/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */
/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include "pointcloudmapping.h"
#include <KeyFrame.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include "Converter.h"
#include <pcl/io/pcd_io.h>
#include <Timer.h>



const int BOUNDARY_MARGIN = 0;
const double DISPLACEMENT_THRESHOLD = 0.01; // 位移阈值(米)

    //Timer timer;
   // timer.Start("Dense Map Construction");  // 开始计时，标签为“稠密地图构建”

PointCloudMapping::PointCloudMapping(double resolution_)
{
    this->resolution = resolution_;
    voxel.setLeafSize( resolution, resolution, resolution);
    globalMap = boost::make_shared< PointCloud >( );
    
    viewerThread = make_shared<thread>( bind(&PointCloudMapping::viewer, this ) );
}

void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    viewerThread->join();
}

void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    cout<<"receive a keyframe, id = "<<kf->mnId<<endl;
    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push_back( kf );
    colorImgs.push_back( color.clone() );
    depthImgs.push_back( depth.clone() );
    
    keyFrameUpdated.notify_one();
}

pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    // 定义在类内部的检测框结构体
    struct Sen {
        double left, top, right, bottom;
        int img_height; // 存储图像高度
    
    Sen() : img_height(0) {}
        // 计算四个边中点坐标
        cv::Point2f getTopMid() const { return cv::Point2f((left + right)/2, top); }
        cv::Point2f getBottomMid() const { return cv::Point2f((left + right)/2, bottom); }
        cv::Point2f getLeftMid() const { return cv::Point2f(left, (top + bottom)/2); }
        cv::Point2f getRightMid() const { return cv::Point2f(right, (top + bottom)/2); }

        // 判断点是否在检测框内（含边界扩展）
        bool isPointInBox(const cv::Point2f& pt) const {
        const float depth_scale = (img_height > 0) ? pt.y/img_height : 0;
        const float margin = BOUNDARY_MARGIN * (0.5 + 0.5*depth_scale); // 近处扩展更大
            return (pt.x >= (left - margin)) && 
                   (pt.x <= (right + margin)) &&
                   (pt.y >= (top - margin)) && 
                   (pt.y <= (bottom + margin));
        }
        
        // 判断是否包含另一个框的中点
        int countContainedMidPoints(const Sen& other) const {
            int count = 0;
            if (isPointInBox(other.getTopMid())) count++;
            if (isPointInBox(other.getBottomMid())) count++;
            if (isPointInBox(other.getLeftMid())) count++;
            if (isPointInBox(other.getRightMid())) count++;
            return count;
        }
    };

    PointCloud::Ptr tmp(new PointCloud());
    Eigen::Isometry3d current_pose = ORB_SLAM2::Converter::toSE3Quat(kf->GetPose());
    
    // 第一次遍历：收集所有类别1和类别3的框
    vector<Sen> class1_boxes;
    vector<Sen> class3_boxes;
    
    for (int k = 0; k < kf->objects_cur_.size(); ++k) {
        int obj_class = kf->objects_cur_[k]->ndetect_class;
        vector<double> box = kf->objects_cur_[k]->vdetect_parameter;
        
        Sen current_box;
        current_box.left = box[0];
        current_box.top = box[1];
        current_box.right = box[2];
        current_box.bottom = box[3];
        
        if (obj_class == 1) {
            class1_boxes.push_back(current_box);
        } 
        
        if (obj_class == 3) {
            class3_boxes.push_back(current_box);
        }
    }
    
    for (int m = 0; m < depth.rows; m += 3) {
        for (int n = 0; n < depth.cols; n += 3) {
            bool inbox = false;
            
          //  int obj_class = kf->objects_cur_[k]->ndetect_class;
            cv::Point2f current_point(n, m);
            
       // 检查点是否在任何类别1的框内
            bool in_class1 = false;
            bool in_class3 = false;
            
            
            for (const auto& box : class1_boxes) {
                if (box.isPointInBox(current_point)) {
                    in_class1 = true;
                    break;
                } 
            }
            
            for (const auto& box : class3_boxes) {
                if (box.isPointInBox(current_point)) {
                    in_class3 = true;
                    break;
                }
            }
            //在3不在1删除
            if (in_class3 && !in_class1)
            {
            inbox =true;         
            }
            //在1不在3保留
            if (in_class1 && !in_class3)
            {
            inbox =false;
            }
            //即不再1也不在3保留
            if (!in_class1 && !in_class3)
            {
            inbox =false;
            }
            //即在1又在3
            if (in_class3 && in_class1)
            {
             
            if(1) {  
             float d = depth.ptr<float>(m)[n];
                if (d < 0.01 || d > 10) continue;
                
                PointT p_current;
                p_current.z = d;
                p_current.x = (n - kf->cx) * p_current.z / kf->fx;
                p_current.y = (m - kf->cy) * p_current.z / kf->fy;
                
                 // 简单匹配：在同一像素位置找上一帧的点
                if (mLastPointCloud && mLastDepth.rows == depth.rows && mLastDepth.cols == depth.cols) {
                    float d_prev = mLastDepth.ptr<float>(m)[n];
                    if (d_prev >= 0.01 && d_prev <= 10) {
                        PointT p_previous;
                        p_previous.z = d_prev;
                        p_previous.x = (n - kf->cx) * p_previous.z / kf->fx;
                        p_previous.y = (m - kf->cy) * p_previous.z / kf->fy;
                        
                        // 转换到世界坐标系
                        Eigen::Vector3d P_world_prev = mLastPose * 
                            Eigen::Vector3d(p_previous.x, p_previous.y, p_previous.z);
                        Eigen::Vector3d P_world_curr = current_pose * 
                            Eigen::Vector3d(p_current.x, p_current.y, p_current.z);
                            
                        // 计算位移
                        double displacement = (P_world_curr - P_world_prev).norm();
                        
                        if (displacement > DISPLACEMENT_THRESHOLD) {
                            continue; // 动态点，跳过
                        }
                    }
                }
            
            }
            
            
         }
            
            
            /***
                //在1框不在3框
                if (in_class1 == true && in_class3== false) {
                 
                // 点在类别1框内 - 保留它
                inbox = false;
                 break;
                 }
                 
               //在3框不在1框
                if(in_class1 == false && in_class3 == true){
                 
                //点在类别3框内-删除它
                inbox=true;
                break;
                }
                
                //既不再1也不在3
                if(in_class1 == false && in_class3 == false){
                 
                //保留它
                inbox = false;
                break;
                }
             
                //即在1也在3
             if(0) {  if(in_class1 == true && in_class3 == true)
                {//3包括1
                for (const auto& box3 : class3_boxes) {
                    if (box3.isPointInBox(current_point)) {
                        // 3包含1数量
                        bool has_overlap = false;
                        int max_count31 = 0;
                        
                        for (const auto& box1 : class1_boxes) {
                            int count = box3.countContainedMidPoints(box1);
                            if (count > 0) {
                                has_overlap = true;
                                if (count > max_count31) {
                                    max_count31 = count;
                                }
                            }
                        }

                        if (has_overlap) {
                        //重叠中点为1
                            if (max_count31 == 1) {
                            //该点保留
                                inbox = true;
                                }
                                //重叠中点为2或4
                            if (max_count31 == 2 || max_count31 == 3 || max_count31 ==4) {
                                //删除
                                inbox = false;
                            }
                        break;
                    }
            }
                }
                
                }
            }
            
           
             if(0) {  if(in_class1 == true && in_class3 == true)
                {
                for (const auto& box1 : class1_boxes) {
                    if (box1.isPointInBox(current_point)) {
                        // 1包含3数量
                        bool has_overlap = false;
                        int max_count13 = 0;
                        
                        for (const auto& box3 : class3_boxes) {
                            int count = box1.countContainedMidPoints(box3);
                            if (count > 0) {
                                has_overlap = true;
                                if (count > max_count13) {
                                    max_count13 = count;
                                }
                            }
                        }

                        if (has_overlap) {
                        //重叠中点为1
                            if (max_count13 == 1) {
                            //该点保留
                                inbox = true;
                                }
                                //重叠中点为2或4
                            if (max_count13 == 2 || max_count13 == 3 || max_count13 ==4) {
                                //删除
                                inbox = false;
                            }
                        break;
                    }
            }
                }
                
                }
            }
*****/
         
            
            if (inbox){
             continue;
           }
                
            
            
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d > 10)
                continue;
                
            PointT p;
            p.z = d;
            p.x = (n - kf->cx) * p.z / kf->fx;
            p.y = (m - kf->cy) * p.z / kf->fy;
            
            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];
                
            tmp->points.push_back(p);
        }
    }
    
    // 保存当前帧数据供下一帧使用
    mLastPointCloud = tmp;
    mLastPose = current_pose;
    mLastDepth = depth.clone(); // 保存深度图
    
    
    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat(kf->GetPose());
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud(*tmp, *cloud, T.inverse().matrix());
    cloud->is_dense = false;
    
    cout << "generate point cloud for kf " << kf->mnId << ", size=" << cloud->points.size() << endl;
    return cloud;

//timer.End();  // 结束计时，自动打印耗时并写入日志
    
}


void PointCloudMapping::viewer()
{
    
    pcl::visualization::CloudViewer viewer("viewer");
    while(1)
    {
        {
            unique_lock<mutex> lck_shutdown(shutDownMutex);
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated(keyFrameUpdateMutex);
            keyFrameUpdated.wait(lck_keyframeUpdated);
        }
        
        // keyframe is updated 
        size_t N=0;
        {
            unique_lock<mutex> lck(keyframeMutex);
            N = keyframes.size();
        }
        
        for (size_t i=lastKeyframeSize; i<N; i++)
        {
            PointCloud::Ptr p = generatePointCloud(keyframes[i], colorImgs[i], depthImgs[i]);
            *globalMap += *p;
        }
        pcl::io::savePCDFileBinary("vslam.pcd", *globalMap);
        
        PointCloud::Ptr tmp(new PointCloud());
        voxel.setInputCloud(globalMap);
        voxel.filter(*tmp);
        globalMap->swap(*tmp);
        
        DIR *dir;
        string save_path = "/home/zs/YOLOV9_ORB_SLAM/runs/zs.pcd";
        pcl::io::savePCDFile(save_path, *tmp);
        viewer.showCloud(globalMap);
        cout<<"show global map, size="<<globalMap->points.size()<<endl;
        lastKeyframeSize = N;
    }
}




