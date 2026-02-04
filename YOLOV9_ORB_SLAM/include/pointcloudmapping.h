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

#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include "System.h"
#include "Frame.h"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <condition_variable>

using namespace ORB_SLAM2;

class PointCloudMapping
{
public:
    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    
    PointCloudMapping( double resolution_ );
    
    // 插入一个keyframe，会更新一次地图
    void insertKeyFrame( KeyFrame* kf, cv::Mat& color, cv::Mat& depth );
    void shutdown();
    void viewer();
    

    
    
    struct Fireword {
    double x, y;
    Fireword(double x, double y);
    bool isInside(double left, double right, double top, double bottom) const;
};

    struct Sen {
    double left, top, right, bottom;
    
    int img_height;
        
        Sen() : img_height(0) {}
    Fireword topMid() const;
    Fireword bottomMid() const;
    Fireword leftMid() const;
    Fireword rightMid() const;
};

private:
    // 新增：存储上一帧的点云和位姿
    pcl::PointCloud<PointT>::Ptr mLastPointCloud;
    Eigen::Isometry3d mLastPose;
    double mDisplacementThreshold = 0.1; // 位移阈值，单位：米
    cv::Mat mLastDepth;  // 上一帧的深度图
    
    
bool isInDynamicCoreRegion(const cv::Point2f& pt, const std::vector<Sen>& dyn_boxes) const;
    
protected:
    PointCloud::Ptr generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth);
    bool isPointFiltered(const Fireword& pt, const Sen& sen1, const Sen& sen3);
    PointCloud::Ptr globalMap;
    shared_ptr<thread>  viewerThread;   
    
    bool    shutDownFlag    =false;
    mutex   shutDownMutex;  
   
    condition_variable  keyFrameUpdated;
    mutex               keyFrameUpdateMutex;
    
    // data to generate point clouds
    vector<KeyFrame*>       keyframes;
    vector<cv::Mat>         colorImgs;
    vector<cv::Mat>         depthImgs;
    mutex                   keyframeMutex;
    uint16_t                lastKeyframeSize =0;
    
    double resolution = 0.04;
    pcl::VoxelGrid<PointT>  voxel;
};

#endif // POINTCLOUDMAPPING_H
