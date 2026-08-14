#include "swf.h"
#include "../utility/visualization.h"
#include<fstream>
#include "../factor/initial_factor.h"

//对窗口内每一帧初始化陀螺偏置并估计初始机身朝向（旋转矩阵），用于后续视觉-惯性初始化
void SWFOptimization::InitializePos() {
    //将每个时间帧的陀螺偏置 Bgs 置为 0
    for (int i = 0; i < SWF_WINDOW_SIZE + 1; i++)
        Bgs[i].setZero();

    // 打印初始 bias 与平均加速度 acc_mean，便于调试/观测
    LOG_OUT << "initial bgs:" << Bgs[0];
    printf("averge acc %f %f %f\n", acc_mean.x(), acc_mean.y(), acc_mean.z());

    // 构造 mag_mean 为 (0,1,0) —— 表示一个假设的磁矢量（仅用于给定参考水平方向）
    Vector3d mag_mean;
    mag_mean(0) = 0;
    mag_mean(1) = 1;
    mag_mean(2) = 0;

    Matrix3d Rwb0;
    // 以加速度平均值方向作为 z 轴（重力方向），即机身在世界坐标系中“朝下/朝上”的方向
    Eigen::Vector3d z0 = acc_mean.normalized();
    // 用磁矢量与 z0 的叉乘（通过反对称矩阵实现）得到一个水平向量并归一化，作为 x 轴方向
    Eigen::Vector3d x0 = (Utility::skewSymmetric(mag_mean) * z0).normalized();
    // 通过 z0 与 x0 的叉乘得到 y 轴，保证右手坐标系且正交
    Eigen::Vector3d y0 = (Utility::skewSymmetric(z0) * x0).normalized();
    // 构造旋转矩阵
    Rwb0.block(0, 0, 1, 3) = x0.transpose();
    Rwb0.block(1, 0, 1, 3) = y0.transpose();
    Rwb0.block(2, 0, 1, 3) = z0.transpose();


    // 将窗口内每帧的初始旋转 Rs 全部设为这个 Rwb0
    for (int i = 0; i < SWF_WINDOW_SIZE + 1; i++)
        Rs[i] = Rwb0;
    LOG_OUT << "init R0: " << endl
            << Utility::R2ypr(Rs[0]).transpose() << ","
            << Utility::R2ypr(Rwb0).transpose() << ","
            << endl;
}


//接收并预处理来自 IMU 的线加速度和角速度数据，然后按时序把样本存入缓冲队列以供后续处理
void SWFOptimization::InputIMU(double t, const Vector3d& linearAcceleration, const Vector3d& angularVelocity) {
    if (first_observe_time == 0)
        first_observe_time = t;
    //跳过初始一段时间内的数据（常用于滤除启动抖动或静止期）
    if (t < first_observe_time + SKIP_TIME)
        return;
    //跳过期之后再累计一段时间的数据用于求平均重力向量（重力方向估计）
    if (t < first_observe_time + SKIP_TIME + AVERAGE_TIME) {
        acc_mean += linearAcceleration;
        acc_count++;
        return;
    }
    static bool imu_initialize = false;
    if (!imu_initialize) {
        imu_initialize = true;
        acc_mean /= acc_count;
        InitializePos();
    }

    acc_buf.push(make_pair(t, linearAcceleration));
    gyr_buf.push(make_pair(t, angularVelocity));
}


bool SWFOptimization::GetImuInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>>& accVector,
                                     vector<pair<double, Eigen::Vector3d>>& gyrVector) {
    if (acc_buf.empty()) {
        printf("not receive imu\n");
        return false;
    }
    if (t1 <= acc_buf.back().first) {
        while (acc_buf.front().first <= t0) {
            acc_buf.pop();
            gyr_buf.pop();
        }
        while (acc_buf.front().first < t1) {
            accVector.push_back(acc_buf.front());
            acc_buf.pop();
            gyrVector.push_back(gyr_buf.front());
            gyr_buf.pop();
        }
        accVector.push_back(acc_buf.front());
        gyrVector.push_back(gyr_buf.front());
    } else {
        printf("wait for imu\n");
        return false;
    }
    return true;
}


bool SWFOptimization::ImuAvailable(double t) {

    double tend = acc_buf.back().first;
    if (!acc_buf.empty() && t <= tend)
        return true;
    else
        return false;
}


void SWFOptimization::ImuIntegrate() {
    if (AMP) {
        static double ACC_N0 = ACC_N, INT_N0 = INT_N, ACC_W0 = ACC_W, GYR_N0 = GYR_N, GYR_W0 = GYR_W;
        if (image_count < SWF_WINDOW_SIZE / AMP_NUM) {
            ACC_N = ACC_N0 * AMP;
            ACC_W = ACC_W0 * AMP;
            GYR_N = GYR_N0 * AMP;
            GYR_W = GYR_W0 * AMP;
        } else {
            static bool is_first = true;
            if (is_first) {
                is_first = false;
                ACC_N = ACC_N0; INT_N = INT_N0; ACC_W = ACC_W0; GYR_N = GYR_N0; GYR_W = GYR_W0;
                for (int i = 0; i < (int)pre_integrations.size(); i++) {
                    if (pre_integrations[i]) {
                        pre_integrations[i]->noise = Eigen::Matrix<double, 18, 18>::Zero();
                        pre_integrations[i]->noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
                        pre_integrations[i]->noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
                        pre_integrations[i]->noise.block<3, 3>(6, 6) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
                        pre_integrations[i]->noise.block<3, 3>(9, 9) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
                        pre_integrations[i]->noise.block<3, 3>(12, 12) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
                        pre_integrations[i]->noise.block<3, 3>(15, 15) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
                        pre_integrations[i]->repropagate(Bas[i], Bgs[i], acc_scale, gyr_scale);
                    }

                }
                if (visual_inertial_bases_global[0]) visual_inertial_bases_global[0]->ResetInit();
            }
        }
    }
    vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;

    GetImuInterval(prev_time, cur_time, accVector, gyrVector);

    ASSERT(accVector.size() >= 1);
    ASSERT(prev_time == prev_time2);
    ASSERT(cur_time >= prev_time);

    ASSERT(accVector.size() > 2);
    double last_time_stamp = accVector[accVector.size() - 1].first;
    double second_last_time_stamp = accVector[accVector.size() - 2].first;
    int acc_size = accVector.size();
    if (abs(last_time_stamp - cur_time) > abs(second_last_time_stamp - cur_time))
        acc_size -= 1;
    Ps[image_count - 1] += Vs[image_count - 1] * old_time_shift;
    Rs[image_count - 1] *= Utility::deltaQ((gyr_0 - Bgs[image_count - 1]) * old_time_shift).toRotationMatrix();
    static double old_imu_time = -1;
    for (int i = 0; i < acc_size; i++) {
        double t = accVector[i].first;
        if (old_imu_time < 0) old_imu_time = t;
        double dt = t - old_imu_time;
        ASSERT(dt >= 0);
        old_imu_time = t;
        headers[image_count - 1] = t;
        if (dt > 0)IMUProcess(dt, accVector[i].second, gyrVector[i].second);
    }
    time_shifts[image_count - 1] = old_imu_time - cur_time;
    if (time_shifts[image_count - 1] == 0)time_shifts[image_count - 1] = 1e-10;
    ASSERT(time_shifts[image_count - 1] != 0);
    old_time_shift = time_shifts[image_count - 1];
    Ps[image_count - 1] -= Vs[image_count - 1] * old_time_shift;
    Rs[image_count - 1] *= Utility::deltaQ(-(gyr_0 - Bgs[image_count - 1]) * old_time_shift).toRotationMatrix();

    prev_time = cur_time;

}

void SWFOptimization::IMUProcess( double dt, const Vector3d& linear_acceleration, const Vector3d& angular_velocity) {
    static bool first_imu = true;
    if (first_imu) {
        first_imu = false;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }
    if (!pre_integrations[image_count - 1]) {
        pre_integrations[image_count - 1] =
            new IntegrationBase{acc_0, gyr_0, Bas[image_count  - 1], Bgs[image_count - 1], acc_scale, gyr_scale};
    }
    if (image_count - 1 != 0) {
        pre_integrations[image_count - 1]->push_back(dt, linear_acceleration, angular_velocity);
        dt_buf[image_count - 1].push_back(dt);
        linear_acceleration_buf[image_count - 1].push_back(linear_acceleration);
        angular_velocity_buf[image_count - 1].push_back(angular_velocity);
        int j = image_count - 1;
        Vector3d un_acc_0 = Rs[j] * ((acc_0.array() * acc_scale.array()).matrix() - Bas[j]) -  G;
        Vector3d un_gyr = 0.5 * ((gyr_0 + angular_velocity).array() * gyr_scale.array()).matrix() - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * ((linear_acceleration.array() * acc_scale.array()).matrix() - Bas[j]) - G;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;

    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;

}






void SWFOptimization::InitializeSqrtInfo() {


    Vector2Double();
    MarginalizationInfo* marginalization_info = new MarginalizationInfo();

    {
        Eigen::Matrix<double, 6, 6>sqrt_info_pose;
        sqrt_info_pose.setZero();
        sqrt_info_pose.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity() * 1e3;
        sqrt_info_pose.block<3, 3>(3, 3) = Eigen::Matrix<double, 3, 3>::Identity() * 1e3;
        ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(
            new InitialPoseFactor(Ps[0] + Rs[0]*TIC[0], Quaterniond(Rs[0]*RIC[0]), sqrt_info_pose),
            0, std::vector<double*> {para_pose[0]}, std::vector<int> {}, std::vector<int> {});
        marginalization_info->addResidualBlockInfo(residual_block_info);
    }

    {
        Eigen::Matrix<double, 9, 9>sqrt_info_bias;
        sqrt_info_bias.setZero();
        sqrt_info_bias.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity() * 1e1;
        sqrt_info_bias.block<3, 3>(3, 3) = Eigen::Matrix<double, 3, 3>::Identity() * 1;
        sqrt_info_bias.block<3, 3>(6, 6) = Eigen::Matrix<double, 3, 3>::Identity() * 1e1;
        ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(
            new InitialBiasFactor(Vs[0], Bas[0], Bgs[0], sqrt_info_bias),
            0, std::vector<double*> {para_speed_bias[0]}, std::vector<int> {}, std::vector<int> {});
        marginalization_info->addResidualBlockInfo(residual_block_info);
    }
    if (ESTIMATE_EXTRINSIC) {
        Eigen::Matrix<double, 6, 6>sqrt_info_pose;
        sqrt_info_pose.setZero();
        sqrt_info_pose.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity() * 100;
        sqrt_info_pose.block<3, 3>(3, 3) = Eigen::Matrix<double, 3, 3>::Identity() * 100;
        ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(
            new InitialPoseFactor(TIC[0], QIC[0], sqrt_info_pose),
            0, std::vector<double*> {para_extrinsic}, std::vector<int> {}, std::vector<int> {});
        marginalization_info->addResidualBlockInfo(residual_block_info);
    }
    if (ESTIMATE_ACC_SCALE) {
        Eigen::Matrix<double, 3, 3>sqrt_info_imu_scale;
        sqrt_info_imu_scale.setZero();
        sqrt_info_imu_scale.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity() * 200;
        ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(
            new InitialFactor33(Eigen::Vector3d({1, 1, 1}), sqrt_info_imu_scale),
            0, std::vector<double*> {acc_scale.data()}, std::vector<int> {}, std::vector<int> {});
        marginalization_info->addResidualBlockInfo(residual_block_info);
    }
    if (ESTIMATE_GYR_SCALE) {
        Eigen::Matrix<double, 3, 3>sqrt_info_imu_scale;
        sqrt_info_imu_scale.setZero();
        sqrt_info_imu_scale.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity() * 200;
        ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(
            new InitialFactor33(Eigen::Vector3d({1, 1, 1}), sqrt_info_imu_scale),
            0, std::vector<double*> {gyr_scale.data()}, std::vector<int> {}, std::vector<int> {});
        marginalization_info->addResidualBlockInfo(residual_block_info);
    }


    marginalization_info->marginalize(true);
    marginalization_info->getParameterBlocks();
    if (last_marg_info) delete last_marg_info;
    last_marg_info = marginalization_info;

}
