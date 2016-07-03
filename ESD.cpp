/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2014-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

// Stdlib
#include <cstdlib>
#include <climits>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cstdint>

#include <boost/format.hpp>


// PCL input/output
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/io/png_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/file_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/registration/ndt.h>
#include <pcl/features/normal_3d.h>
 
//PCL other
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/supervoxel_clustering.h>

// The segmentation class this example is for
#include <pcl/segmentation/lccp_segmentation.h>

// VTK
#include <vtkImageReader2Factory.h>
#include <vtkImageReader2.h>
#include <vtkImageData.h>
#include <vtkImageFlip.h>
#include <vtkPolyLine.h>

 //planar segmentation
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

//time usage
#include <ctime>

#define CURVATURE UINT32_MAX
#define PLANE 0

/// *****  Type Definitions ***** ///

typedef pcl::PointXYZRGBA PointT;  // The point type used for input
typedef pcl::PointCloud<PointT> PointCloudT; // used when showing supervoxel 
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointNCloudT;
typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;
typedef pcl::LCCPSegmentation<PointT>::SupervoxelAdjacencyList SuperVoxelAdjacencyList; //the adjacency recorded data types

/// Callback and variables

bool show_normals = false, normals_changed = false;
bool show_adjacency = false;
bool show_supervoxels = false;
bool show_help = true;
double coefficients_x, coefficients_y, coefficients_z, coefficients_num;

//todo
float voxel_resolution;//0.05 ,0.008
float seed_resolution;//0.2  ,0.032
float color_importance = 0.1f;
float spatial_importance = 1.0f;
float normal_importance = 4.0f;
bool use_single_cam_transform = false;

// MPSS parameter
//double pre_filter_threshold = 0.99;
//Between supervoxels
double parrallel_threshold;//0.8  //dot_product , threshold to consider as parrallel
// Between Surface
double parrallel_filter;
double distance_to_plane; //0.005,0.08
double curvature_ratio = 100;//todo
//double planes_difference = 0.1;
//int    remain_ratio = 20;

//Agglomerative Surface Growing Learning Rate
double mu;

// global
std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
std::map<uint32_t, int> clusters_int;
std::map<uint32_t, bool> clusters_used;
std::vector<uint32_t> plane;
std::vector< std::vector<uint32_t> > planesVectors;
std::vector<size_t> orderVectors;// Remember index of planesVectors to descending order
std::vector<double> aver_nor_x,aver_nor_y,aver_nor_z; 
std::vector<double> aver_pos_x,aver_pos_y,aver_pos_z; 
std::vector<float> aver_var;
std::map<uint32_t, uint32_t> sv_label_to_seg_label_map;

std::vector<double> normal_vector_x,normal_vector_y,normal_vector_z;
std::vector<double> pos_x, pos_y, pos_z;
float max_x=0, max_y=0, max_z=0;
float min_x=0, min_y=0, min_z=0;
int size_temp=0;
double avn_x=0,avn_y=0,avn_z=0;
double avp_x=0, avp_y=0, avp_z=0;

// handle the vtk stuff of visualization
void addSupervoxelConnectionsToViewer (PointT &supervoxel_center,
                                       PointCloudT &adjacent_supervoxel_centers,
                                       std::string supervoxel_name,
                                       boost::shared_ptr<pcl::visualization::PCLVisualizer> & viewer);

void savePCDfile(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const char* fileName);
void findNeighbor(std::vector<uint32_t>& plane,const uint32_t& the_cluster_num
                  , double& the_normal_x, double& the_normal_y, double& the_normal_z);
void scaleAddCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr add_cloud_ptr, double plane_variance, double scale_ratio);
size_t findProjectPoint(float x, float y, float z, pcl::PointCloud<pcl::PointXYZRGB>::Ptr augment_cloud, size_t AR_planar_label, float& new_x ,float& new_y, float& new_z);
void replaceRGB_AR(pcl::PointCloud<pcl::PointXYZRGB>::Ptr augment_cloud, pcl::PointCloud<pcl::PointXYZRGBL>::Ptr result_cloud_ptr, size_t AR_planar);
bool loadPointCloudFile(const std::string& fileName, pcl::PCLPointCloud2& pointCloud);


/// ---- main ---- ///
int
main (int argc,
      char ** argv)
{
  if (argc < 6) {
    PCL_INFO("Usage: ./ESD [supervoxel_scale] [input_point_cloud] [parrallel_threshold] [mu] [parrallel_filter] [distance_to_plane] (-sr) (-o [save_filename]) (-apc [aug_point_cloud])\n");
    PCL_INFO("  Ex:  ./ESD 0.00568 my.pcd(.ply) 0.8 0.2 0.8 0.005 \n");
    PCL_INFO("  Ex:  ./ESD 0.00568 my.pcd(.ply) 0.8 0.2 0.8 0.005 -sr\n");
    PCL_INFO("  Ex:  ./ESD 0.00568 my.pcd(.ply) 0.8 0.2 0.8 0.005 -apc my_pic.ply\n");
    PCL_INFO("  Ex:  ./ESD 0.00568 my.pcd(.ply) 0.8 0.2 0.8 0.005 -sr -apc my_pic.ply\n");
    PCL_INFO("  Ex:  ./ESD 0.00568 my.pcd(.ply) 0.8 0.2 0.8 0.005 -sr -o saved.ply -apc my_pic.ply\n");
    PCL_INFO("Notice:\n");
    PCL_INFO("  [input_point_cloud] and [aug_point_cloud] supports .ply and .pcd\n");
    PCL_INFO("  format of [save_filename] is \"xyzrgbl\"\n");
    PCL_INFO("  -sr: show result\n");
    PCL_INFO("  -apc: augment point cloud\n");
    return false;
  }

  /// -----------------------------------|  Preparations  |-----------------------------------

  bool add_label_field = true;
  bool show_svcloud = pcl::console::find_switch (argc, argv, "-sv");
  bool show_result = pcl::console::find_switch (argc, argv, "-sr");
  bool aug_pc = pcl::console::find_switch (argc, argv, "-apc");
  bool save_pc = pcl::console::find_switch (argc, argv, "-o");
  /// Create variables needed for preparations

  //outputname can be changed customly
  pcl::PointCloud<PointT>::Ptr input_cloud_ptr (new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::Normal>::Ptr input_normals_ptr (new pcl::PointCloud<pcl::Normal>); 
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr add_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr augment_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr RANSAC_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr add_cloud_ptr2 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr augment_cloud2 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr RANSAC_cloud2 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr add_cloud_ptr3 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr augment_cloud3 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr RANSAC_cloud3 (new pcl::PointCloud<pcl::PointXYZRGB>);
  
  bool has_normals = false;
  
  /// Get pcd path from command line
  std::string pcd_filename = argv[2];
  PCL_INFO ("Loading pointcloud\n");
  float supervoxel_scale = atof(argv[1]);
  double ransacThreshold = 0.001;
  parrallel_threshold = atof(argv[3]);
  mu = atof(argv[4]);
  parrallel_filter = atof(argv[5]);
  distance_to_plane = atof(argv[6]);
  
  /// check if the provided pcd file contains normals
  pcl::PCLPointCloud2 input_pointcloud2;  //inpu_pointcloud2 ,new version of pcl
  if (loadPointCloudFile(pcd_filename, input_pointcloud2))
  {
    PCL_ERROR ("ERROR: Could not read input point cloud %s.\n", pcd_filename.c_str ());
    return (3);
  }
  pcl::fromPCLPointCloud2 (input_pointcloud2, *input_cloud_ptr);
  PCL_INFO ("Done making cloud\n");

  //time usage
  std::clock_t start;
  double duration;
  start = std::clock();
/// -----------------------------------|  Main Computation  |-----------------------------------
  // Default values 
  //finding max & min in the point cloud
  for (size_t i = 0; i != input_cloud_ptr->points.size(); ++i){
    pcl::PointXYZRGBA checkP = input_cloud_ptr->points.at(i);
    if(checkP.x > max_x)
      max_x = checkP.x;
    if(checkP.y > max_y)
      max_y = checkP.y;
    if(checkP.z > max_z)
      max_z = checkP.z;
    if(checkP.x < min_x)
      min_x = checkP.x;
    if(checkP.y < min_y)
      min_y = checkP.y;
    if(checkP.z < min_z)
      min_z = checkP.z;
  }
  //Input point cloud variance calculation, XYZ stuff
  //std::cout<<"MAX:"<<max_x<<","<<max_y<<","<<max_z<<endl;
  //std::cout<<"MIN:"<<min_x<<","<<min_y<<","<<min_z<<endl;
  //std::cout<<"input_dis_var: "<<pow((max_x-min_x)*(max_y-min_y)*(max_z-min_z),1/3.)<<endl;
  //parameter
  double input_dis_var = pow((max_x-min_x)*(max_y-min_y)*(max_z-min_z),1/3.);
  voxel_resolution = supervoxel_scale * input_dis_var;
  seed_resolution = voxel_resolution * 4;
  //std::cout<<"voxel_resolution: "<<voxel_resolution<<endl;
  // Supervoxel Stuff
  /// Preparation of Input: Supervoxel Oversegmentation

  pcl::SupervoxelClustering<PointT> super (voxel_resolution, seed_resolution, use_single_cam_transform);
  super.setUseSingleCameraTransform(use_single_cam_transform);
  super.setInputCloud (input_cloud_ptr);
  if (has_normals)
    super.setNormalCloud (input_normals_ptr);
  super.setColorImportance (color_importance);
  super.setSpatialImportance (spatial_importance);
  super.setNormalImportance (normal_importance);
  std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters;

  PCL_INFO ("Extracting supervoxels\n");
  super.extract (supervoxel_clusters); //initCompute & preapreForSegmentation(generate most variables) ,selectInitialSupervoxelSeeds(start the index mapping using Octree),
  // createSupervoxelHeplers(record all the variables while computing), expandSupervoxels(iterate through each seeds),makeSupervoxels(finally duplicate the computed final version)

  std::stringstream temp;
  temp << "  Nr. Supervoxels: " << supervoxel_clusters.size () << "\n";
  PCL_INFO (temp.str ().c_str ());

  PCL_INFO ("Getting supervoxel adjacency\n");
  super.getSupervoxelAdjacency (supervoxel_adjacency);
  //time usage due
  duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
  std::cerr<<"sv time usage: "<< duration <<'\n';

  //===============================   MPSS parameter evaluation   =================================
  std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr>::iterator label_itr = supervoxel_clusters.begin();
  std::multimap<int,uint32_t> clusters_belong;
  std::multimap<int,uint32_t>::iterator belong_itr = clusters_belong.begin();
  std::map<uint32_t,float> label2norm_x,label2norm_y,label2norm_z;
  std::map<uint32_t,float> label2pos_x,label2pos_y,label2pos_z;
  std::map<uint32_t,float> label2var;
  clusters_int.clear();
  std::vector<int> sizeVectors;

  int i=0;
  float average_nor_x =0.0; double average_nor_y = 0.0; double average_nor_z =0.0;
  for (; label_itr != supervoxel_clusters.end (); label_itr++)
    {
      pcl::Supervoxel<PointT>::Ptr supervoxel = label_itr->second;
      normal_vector_x.push_back(supervoxel->normal_.normal_x);
      normal_vector_y.push_back(supervoxel->normal_.normal_y);
      normal_vector_z.push_back(supervoxel->normal_.normal_z);
      pos_x.push_back(supervoxel->centroid_.x);
      pos_y.push_back(supervoxel->centroid_.y);
      pos_z.push_back(supervoxel->centroid_.z);
      clusters_int.insert(std::pair<uint32_t,int>(label_itr->first,i));
      clusters_used.insert(std::pair<uint32_t,bool>(label_itr->first,false));
      i++;
    }
    float count = float(i);
    count/=3.0;
  average_nor_x /= count; average_nor_y /= count; average_nor_z /= count;

  //===============================   MPSS  =================================
  label_itr = supervoxel_clusters.begin();
  int clusters_belong_int=0;
  int the_cluster_int;
  int size_max = 0;
  int size_temp = 0;
  int neighbor_count = 0;
  double the_normal_x=0, the_normal_y=0, the_normal_z=0; 
  int vector_int = 0;
  uint32_t the_cluster_num;
  int trial = 0;
  //start labeling
  while(label_itr != supervoxel_clusters.end() )
  {
    if( clusters_used.find(label_itr->first)->second==true )
    {
      label_itr++;
      continue;
    }
    clusters_used.find(label_itr->first)->second = true;
    trial++;
    the_cluster_num = label_itr->first;
    the_cluster_int = clusters_int.find(the_cluster_num)->second;
    the_normal_x = normal_vector_x[the_cluster_int];
    the_normal_y = normal_vector_y[the_cluster_int];
    the_normal_z = normal_vector_z[the_cluster_int];
    //std::cerr<<endl<<"x: "<<the_normal_x<<endl<<"y: "<<the_normal_y<<endl<<"z: "<<the_normal_z<<endl;
    plane.clear();
    ::size_temp = 0;
    ::avp_x = 0;
    ::avp_y = 0;
    ::avp_z = 0;
    ::avn_x = 0;
    ::avn_y = 0;
    ::avn_z = 0;
    findNeighbor(plane,the_cluster_num,the_normal_x,the_normal_y,the_normal_z);
    if(::size_temp <= 1)
      continue;
    ::avn_x /= double(::size_temp);
    ::avn_y /= double(::size_temp);
    ::avn_z /= double(::size_temp);
    ::avp_x /= double(::size_temp);
    ::avp_y /= double(::size_temp);
    ::avp_z /= double(::size_temp);
    std::vector<uint32_t>::iterator super_it = plane.begin();
    double var_x=0,var_y=0,var_z=0;
    for(;super_it!=plane.end();super_it++){
      int the_cluster_int = clusters_int.find(*super_it)->second;
      var_x += pow(normal_vector_x[the_cluster_int]-avn_x,2);
      var_y += pow(normal_vector_y[the_cluster_int]-avn_y,2);
      var_z += pow(normal_vector_z[the_cluster_int]-avn_z,2);
    }
    double var = (var_x+var_y+var_z)/double(::size_temp);
    //std::cout<<"Var x:"<<var_x<<",y:"<<var_y<<",z:"<<var_z<<endl;
    //std::cout<<"Variance"<<var<<endl;
    if(var>=0.1){
      bool newCurve = true;
      //Curvature Refinements
      for(std::vector<double>::iterator find_it = aver_pos_x.begin();find_it!=aver_pos_x.end();find_it++){
        size_t diff = find_it-aver_pos_x.begin();
        double op_x = aver_pos_x[diff];
        double op_y = aver_pos_y[diff];
        double op_z = aver_pos_z[diff];
        if(planesVectors[diff][0]==CURVATURE && std::abs(avp_x-op_x)+std::abs(avp_y-op_y)+std::abs(avp_z-op_z) < 
            (std::abs(max_x-min_x)+std::abs(max_y-min_y)+std::abs(max_z-min_z))/curvature_ratio ){
          newCurve = false;
          planesVectors[diff].insert(planesVectors[diff].end(),plane.begin(),plane.end());
          break;
        }
      }
      if(newCurve==true){
        std::vector<uint32_t> tmp(1,CURVATURE);
        tmp.insert(tmp.end(),plane.begin(),plane.end());
        planesVectors.push_back(tmp);
        aver_nor_x.push_back(avn_x);
        aver_nor_y.push_back(avn_y);
        aver_nor_z.push_back(avn_z);
        aver_pos_x.push_back(avp_x);
        aver_pos_y.push_back(avp_y);
        aver_pos_z.push_back(avp_z);
        aver_var.push_back(var);
      }
    }
    //else{
      bool new_plane = true;
      //Planar Refinements
      for(std::vector<double>::iterator find_it = aver_nor_x.begin();find_it!=aver_nor_x.end();find_it++){
        size_t diff = find_it-aver_nor_x.begin();
        double on_x = *find_it;
        double on_y = aver_nor_y[diff];
        double on_z = aver_nor_z[diff];
        double op_x = aver_pos_x[diff];
        double op_y = aver_pos_y[diff];
        double op_z = aver_pos_z[diff];
        if(planesVectors[diff][0]==PLANE && std::abs(avn_x*on_x +avn_y*on_y +avn_z*on_z) > parrallel_filter && 
          std::abs((avn_x*avp_x + avn_y*avp_y + avn_z*avp_z) - (on_x*op_x + on_y*op_y + on_z*op_z)) < distance_to_plane ) {
          new_plane = false;
          double weight = plane.size() / double(plane.size()+planesVectors[diff].size());
          aver_nor_x[diff] = (1-weight)*aver_nor_x[diff] + weight*::avn_x;
          aver_nor_y[diff] = (1-weight)*aver_nor_y[diff] + weight*::avn_y;
          aver_nor_z[diff] = (1-weight)*aver_nor_z[diff] + weight*::avn_z;
          aver_pos_x[diff] = (1-weight)*aver_pos_x[diff] + weight*::avp_x;
          aver_pos_y[diff] = (1-weight)*aver_pos_y[diff] + weight*::avp_y;
          aver_pos_z[diff] = (1-weight)*aver_pos_z[diff] + weight*::avp_z;
          planesVectors[diff].insert(planesVectors[diff].end(),plane.begin(),plane.end());
          break;
        }
      }
      if(new_plane == true){
        std::vector<uint32_t> tmp(1,PLANE);
        tmp.insert(tmp.end(),plane.begin(),plane.end());
        planesVectors.push_back(tmp);
        aver_nor_x.push_back(avn_x);
        aver_nor_y.push_back(avn_y);
        aver_nor_z.push_back(avn_z);
        aver_pos_x.push_back(avp_x);
        aver_pos_y.push_back(avp_y);
        aver_pos_z.push_back(avp_z);
        aver_var.push_back(var);
      }
    //}

  }
  //time usage due
  duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
  std::cerr<<"time usage: "<< duration <<'\n';

  pcl::PointXYZRGBA the_point;
  pcl::PointCloud<pcl::PointXYZRGBA> the_points;
  pcl::PointCloud<PointT>::Ptr planes_grown(new pcl::PointCloud<PointT>);
  std::vector<std::vector<uint32_t> >::iterator plane_it = planesVectors.begin();
  std::vector<size_t>::iterator order_it = orderVectors.begin();
  int saving_num = 0;
  pcl::PointCloud<pcl::PointXYZL>::Ptr sv_labeled_cloud = super.getLabeledCloud ();
  pcl::PointCloud<pcl::PointXYZL>::Ptr mpss_labeled_cloud = sv_labeled_cloud->makeShared ();

  //uint32_t curveNo = 2;
  uint32_t planeNo = 2;
  int sizetemp = 0;
  size_t max_planar =plane_it - planesVectors.begin();
  // For finding max & relabel XYZL for visualization
  for(;plane_it != planesVectors.end(); plane_it++){
    // ================== relabel ================
    std::vector<uint32_t>::iterator it = plane_it->begin();
    if(*it==CURVATURE){
      size_t diff = plane_it-planesVectors.begin();
      label2norm_x.insert(std::pair<uint32_t,float>(planeNo,aver_nor_x[diff]));
      label2norm_y.insert(std::pair<uint32_t,float>(planeNo,aver_nor_y[diff]));
      label2norm_z.insert(std::pair<uint32_t,float>(planeNo,aver_nor_z[diff]));
      label2pos_x.insert(std::pair<uint32_t,float>(planeNo,aver_pos_x[diff]));
      label2pos_y.insert(std::pair<uint32_t,float>(planeNo,aver_pos_y[diff]));
      label2pos_z.insert(std::pair<uint32_t,float>(planeNo,aver_pos_z[diff]));
      label2var.insert(std::pair<uint32_t,float>(planeNo,aver_var[diff]));
      //std::cout<<"label2norm: No."<<planeNo<<"  "<<diff<<","<<aver_nor_x[diff]<<","<<aver_nor_y[diff]<<","<<aver_nor_z[diff]<<endl;
      for(it = plane_it->begin()+1; it!= plane_it->end(); it++)
        sv_label_to_seg_label_map[*it]=planeNo;
      planeNo++;
    }
    else{
      size_t diff = plane_it-planesVectors.begin();
      label2norm_x.insert(std::pair<uint32_t,float>(planeNo,aver_nor_x[diff]));
      label2norm_y.insert(std::pair<uint32_t,float>(planeNo,aver_nor_y[diff]));
      label2norm_z.insert(std::pair<uint32_t,float>(planeNo,aver_nor_z[diff]));
      label2pos_x.insert(std::pair<uint32_t,float>(planeNo,aver_pos_x[diff]));
      label2pos_y.insert(std::pair<uint32_t,float>(planeNo,aver_pos_y[diff]));
      label2pos_z.insert(std::pair<uint32_t,float>(planeNo,aver_pos_z[diff]));
      label2var.insert(std::pair<uint32_t,float>(planeNo,aver_var[diff]));
      //std::cout<<"label2norm: No."<<planeNo<<"  "<<diff<<","<<aver_nor_x[diff]<<","<<aver_nor_y[diff]<<","<<aver_nor_z[diff]<<endl;
      for(it = plane_it->begin()+1; it!= plane_it->end(); it++)
        sv_label_to_seg_label_map[*it]=planeNo;
      planeNo++;
    }
    // ================== findMax ================
    if(plane_it->size() > sizetemp){
      sizetemp = plane_it->size();
      max_planar = plane_it - planesVectors.begin();
    }
    // ================== orderVectors =======================
    size_t orderNo = plane_it - planesVectors.begin();
    //std::cout<<orderNo<<endl;
    if(orderVectors.size()==0){
      orderVectors.push_back( orderNo );
    }
    else{
      bool atEnd = true;
      for(order_it = orderVectors.begin();order_it!=orderVectors.end(); order_it++){
        if(plane_it->size()>=planesVectors[*order_it].size()){
          orderVectors.insert(order_it,orderNo);
          atEnd = false;
          break;
        }
      }
      if(atEnd == true)
        orderVectors.push_back(orderNo);
    }
  }
  std::cerr<<"on total "<<mpss_labeled_cloud->points.size()<<endl;

  /*for(order_it = orderVectors.begin();order_it!=orderVectors.end(); order_it++){
    std::cout<<order_it-orderVectors.begin()<<":"<<planesVectors[*order_it].size()<<endl;
  }*/
  pcl::PointCloud<pcl::PointXYZL>::iterator voxel_itr = mpss_labeled_cloud->begin ();
  int no = 0;
  for (; voxel_itr != mpss_labeled_cloud->end (); voxel_itr++){
    voxel_itr->label = sv_label_to_seg_label_map[voxel_itr->label];
  }
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr result_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGBL>);
  std::string file_name = ( pcd_filename.erase(0,4) ).erase(size_t(pcd_filename.end()-pcd_filename.begin())-4,4);
  
  //===== save file (format:xyzrgbl)=====
  if( input_cloud_ptr->points.size() != mpss_labeled_cloud->points.size() )
    PCL_ERROR ("ERROR: Size of input point cloud (xyzrgb) != labeld point cloud (xyzl)");
  for(size_t i = 0; i != input_cloud_ptr->points.size(); ++i){
    pcl::PointXYZRGBA inputP = input_cloud_ptr->points.at(i);
    pcl::PointXYZL labelP = mpss_labeled_cloud->points.at(i);
    pcl::PointXYZRGBL newPoint;
    pcl::PointXYZRGB newP;
    newPoint.x = inputP.x;
    newPoint.y = inputP.y;
    newPoint.z = inputP.z;
    newPoint.r = inputP.r;
    newPoint.g = inputP.g;
    newPoint.b = inputP.b;
    uint32_t l = labelP.label;
    newPoint.label = l;
    result_cloud_ptr->points.push_back(newPoint);
    newP.x = inputP.x;
    newP.y = inputP.y;
    newP.z = inputP.z;
    newP.r = inputP.r;
    newP.g = inputP.g;
    newP.b = inputP.b;
    rgb_cloud_ptr->points.push_back(newP);
  }
  if (save_pc){
    std::string save_filename = argv[pcl::console::find_argument(argc, argv, "-o") +1];
    pcl::io::savePLYFileASCII(save_filename+".ply",*result_cloud_ptr);
  }

  if (aug_pc)
  { 
    std::string add_filename = argv[pcl::console::find_argument(argc, argv, "-apc") +1];
    std::string add_filename2 = argv[pcl::console::find_argument(argc, argv, "-apc") +2];
    std::string add_filename3 = argv[pcl::console::find_argument(argc, argv, "-apc") +3];
    PCL_INFO ("Loading file to add\n");

    //=================== reading add point cloud =====================
    pcl::PCLPointCloud2 input_pointcloud_add;  //inpu_pointcloud2 ,new version of pcl
    if (loadPointCloudFile(add_filename, input_pointcloud_add))
    {
      PCL_ERROR ("ERROR: Could not read add point cloud %s.\n", add_filename.c_str ());
      return (3);
    }
    pcl::fromPCLPointCloud2 (input_pointcloud_add, *add_cloud_ptr);
    PCL_INFO ("Done making cloud\n");

    pcl::PCLPointCloud2 input_pointcloud_add2;  //inpu_pointcloud2 ,new version of pcl
    if (loadPointCloudFile(add_filename2, input_pointcloud_add2))
    {
      PCL_ERROR ("ERROR: Could not read add point cloud %s.\n", add_filename2.c_str ());
      return (3);
    }
    pcl::fromPCLPointCloud2 (input_pointcloud_add2, *add_cloud_ptr2);
    PCL_INFO ("Done making cloud\n");

    pcl::PCLPointCloud2 input_pointcloud_add3;  //inpu_pointcloud2 ,new version of pcl
    if (loadPointCloudFile(add_filename3, input_pointcloud_add3))
    {
      PCL_ERROR ("ERROR: Could not read add point cloud %s.\n", add_filename3.c_str ());
      return (3);
    }
    pcl::fromPCLPointCloud2 (input_pointcloud_add3, *add_cloud_ptr3);
    PCL_INFO ("Done making cloud\n");

    //===== transformation =====
    avp_x=0;avp_y=0;avp_z=0;avn_x=0;avn_y=0;avn_z=0;
    for (size_t i = 0; i < add_cloud_ptr->points.size(); i++) {
      avp_x += add_cloud_ptr->points[i].x;
      avp_y += add_cloud_ptr->points[i].y;
      avp_z += add_cloud_ptr->points[i].z;
    }
    avp_x /= double(add_cloud_ptr->points.size());
    avp_y /= double(add_cloud_ptr->points.size());
    avp_z /= double(add_cloud_ptr->points.size());
    avn_x = 1;
    avn_y = 0;
    avn_z = 0;
    
    size_t AR_planar = max_planar;
    double target_pos_x = aver_pos_x[AR_planar];
    double target_pos_y = aver_pos_y[AR_planar];
    double target_pos_z = aver_pos_z[AR_planar];
    double plane_max_x = DBL_MIN;
    double plane_max_y = DBL_MIN;
    double plane_max_z = DBL_MIN;
    double plane_min_x = DBL_MAX;
    double plane_min_y = DBL_MAX;
    double plane_min_z = DBL_MAX;
    for(size_t i = 0; i<result_cloud_ptr->points.size();i++){
      pcl::PointXYZRGBL theP = result_cloud_ptr->points.at(i);
      if(theP.label == AR_planar+2){
        if(theP.x > plane_max_x)
          plane_max_x = theP.x;
        if(theP.y > plane_max_y)
          plane_max_y = theP.y;
        if(theP.z > plane_max_z)
          plane_max_z = theP.z;
        if(theP.x < plane_min_x)
          plane_min_x = theP.x;
        if(theP.y < plane_min_y)
          plane_min_y = theP.y;
        if(theP.z < plane_min_z)
          plane_min_z = theP.z;
        pcl::PointXYZRGB newP;
        newP.x = theP.x;
        newP.y = theP.y;
        newP.z = theP.z;
        newP.r = theP.r;
        newP.g = theP.g;
        newP.b = theP.b;
        RANSAC_cloud->push_back(newP);
      }
    }
    double plane_variance = pow( pow(plane_max_x - plane_min_x,2.)+pow(plane_max_y - plane_min_y,2.)+pow(plane_max_z - plane_min_z,2.) , 0.5);
    scaleAddCloud(add_cloud_ptr,plane_variance,0.45);

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (ransacThreshold);
  
    seg.setInputCloud (RANSAC_cloud);
    seg.segment (*inliers, *coefficients);
  
    if (inliers->indices.size () == 0)
        PCL_ERROR ("Could not estimate a planar model for the given dataset.");
  
    double target_nor_x = coefficients->values[0];
    double target_nor_y = coefficients->values[1];
    double target_nor_z = coefficients->values[2];
    std::cout<<"nor: "<<target_nor_x<<" "<<target_nor_y<<" "<<target_nor_z<<endl;

    
    Eigen::Affine3f trans_matrix = Eigen::Affine3f::Identity();
    // Define a translation of 2.5 meters on the x axis.(x,y,z)
    trans_matrix.translation() << target_pos_x - avp_x + 0.8, target_pos_y - avp_y, target_pos_z - avp_z;
  
    double alpha = acos( (target_nor_y*avn_y + target_nor_z*avn_z) /sqrt(target_nor_y*target_nor_y + target_nor_z*target_nor_z) );
    double beta = acos( (target_nor_z*avn_z + target_nor_x*avn_x)/sqrt(target_nor_x*target_nor_x + target_nor_z*target_nor_z) );
    double gamma = acos( (target_nor_y*avn_y + target_nor_x*avn_x) /sqrt(target_nor_y*target_nor_y + target_nor_x*target_nor_x) );
    //float gamma = acos( (target_nor_x*avn_x + target_nor_y*avn_y)/sqrt(target_nor_x*target_nor_x + target_nor_y*target_nor_y) );
    //std::cout<<alpha<<","<<beta<<"  ;"<<mean<<endl;
  
    trans_matrix.rotate ( Eigen::AngleAxisf (alpha, Eigen::Vector3f::UnitX()) * Eigen::AngleAxisf (beta, Eigen::Vector3f::UnitY()));//* Eigen::AngleAxisf (gamma, Eigen::Vector3f::UnitZ()) );
  
    // Print the transformation
    printf ("\nUsing an Affine3f derive trans_matrix:\n");
    std::cout << trans_matrix.matrix() << std::endl;
  
    pcl::transformPointCloud(*add_cloud_ptr, *augment_cloud, trans_matrix);
    //replaceRGB
    std::cerr<<"plane point:"<<RANSAC_cloud->points.size()<<", with "<<augment_cloud->points.size()<<endl;
    //replaceRGB_AR(augment_cloud,result_cloud_ptr,AR_planar);
    
    // ======================= second =========================
    AR_planar = orderVectors[1];
    target_pos_x = aver_pos_x[AR_planar];
    target_pos_y = aver_pos_y[AR_planar];
    target_pos_z = aver_pos_z[AR_planar];
    plane_max_x = DBL_MIN;
    plane_max_y = DBL_MIN;
    plane_max_z = DBL_MIN;
    plane_min_x = DBL_MAX;
    plane_min_y = DBL_MAX;
    plane_min_z = DBL_MAX;
    for(size_t i = 0; i<result_cloud_ptr->points.size();i++){
      pcl::PointXYZRGBL theP = result_cloud_ptr->points.at(i);
      if(theP.label == AR_planar+2){
        pcl::PointXYZRGB newP;
        if(theP.x > plane_max_x)
          plane_max_x = theP.x;
        if(theP.y > plane_max_y)
          plane_max_y = theP.y;
        if(theP.z > plane_max_z)
          plane_max_z = theP.z;
        if(theP.x < plane_min_x)
          plane_min_x = theP.x;
        if(theP.y < plane_min_y)
          plane_min_y = theP.y;
        if(theP.z < plane_min_z)
          plane_min_z = theP.z;
        newP.x = theP.x;
        newP.y = theP.y;
        newP.z = theP.z;
        newP.r = theP.r;
        newP.g = theP.g;
        newP.b = theP.b;
        RANSAC_cloud2->push_back(newP);
      }
    }
    plane_variance = pow( pow(plane_max_x - plane_min_x,2.)+pow(plane_max_y - plane_min_y,2.)+pow(plane_max_z - plane_min_z,2.) , 0.5);
    scaleAddCloud(add_cloud_ptr2,plane_variance,0.6);

    pcl::ModelCoefficients::Ptr coefficients2 (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers2 (new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZRGB> seg2;
    // Optional
    seg2.setOptimizeCoefficients (true);
    // Mandatory
    seg2.setModelType (pcl::SACMODEL_PLANE);
    seg2.setMethodType (pcl::SAC_RANSAC);
    seg2.setDistanceThreshold (ransacThreshold);
  
    seg2.setInputCloud (RANSAC_cloud2);
    seg2.segment (*inliers2, *coefficients2);
  
    if (inliers2->indices.size () == 0)
        PCL_ERROR ("Could not estimate a planar model for the given dataset.");
  
    target_nor_x = coefficients2->values[0];
    target_nor_y = coefficients2->values[1];
    target_nor_z = coefficients2->values[2];
    std::cout<<"nor: "<<target_nor_x<<" "<<target_nor_y<<" "<<target_nor_z<<endl;
    
    Eigen::Affine3f trans_matrix2 = Eigen::Affine3f::Identity();
    // Define a translation of 2.5 meters on the x axis.(x,y,z)
    trans_matrix2.translation() << target_pos_x - avp_x + 0.12, target_pos_y - avp_y, target_pos_z - avp_z;

    avn_x = 0;
    avn_y = 0;
    avn_z = 1;
  
    alpha = acos( (target_nor_y*avn_y + target_nor_z*avn_z) /sqrt(target_nor_y*target_nor_y + target_nor_z*target_nor_z) );
    beta = acos( (target_nor_z*avn_z + target_nor_x*avn_x)/sqrt(target_nor_x*target_nor_x + target_nor_z*target_nor_z) );
    gamma = acos( (target_nor_y*avn_y + target_nor_x*avn_x) /sqrt(target_nor_y*target_nor_y + target_nor_x*target_nor_x) );
    //float gamma = acos( (target_nor_x*avn_x + target_nor_y*avn_y)/sqrt(target_nor_x*target_nor_x + target_nor_y*target_nor_y) );
    //std::cout<<alpha<<","<<beta<<"  ;"<<mean<<endl;
  
    trans_matrix2.rotate ( Eigen::AngleAxisf (alpha, Eigen::Vector3f::UnitX()) * Eigen::AngleAxisf (beta, Eigen::Vector3f::UnitY())*Eigen::AngleAxisf(-2.0,Eigen::Vector3f::UnitZ()) );//* Eigen::AngleAxisf (gamma, Eigen::Vector3f::UnitZ()) );
  
    // Print the transformation
    printf ("\nUsing an Affine3f derive trans_matrix:\n");
    std::cout << trans_matrix2.matrix() << std::endl;
  
    pcl::transformPointCloud(*add_cloud_ptr2, *augment_cloud2, trans_matrix2);

    //replaceRGB
    std::cerr<<"Total point:"<<augment_cloud2->points.size()<<", fitting "<<RANSAC_cloud2->points.size()<<endl;
    //replaceRGB_AR(augment_cloud2,result_cloud_ptr,AR_planar);

    // ======================= third =========================
    AR_planar = orderVectors[2];
    target_pos_x = aver_pos_x[AR_planar];
    target_pos_y = aver_pos_y[AR_planar];
    target_pos_z = aver_pos_z[AR_planar];
    plane_max_x = DBL_MIN;
    plane_max_y = DBL_MIN;
    plane_max_z = DBL_MIN;
    plane_min_x = DBL_MAX;
    plane_min_y = DBL_MAX;
    plane_min_z = DBL_MAX;
    for(size_t i = 0; i<result_cloud_ptr->points.size();i++){
      pcl::PointXYZRGBL theP = result_cloud_ptr->points.at(i);
      if(theP.label == AR_planar+2){
        if(theP.x > plane_max_x)
          plane_max_x = theP.x;
        if(theP.y > plane_max_y)
          plane_max_y = theP.y;
        if(theP.z > plane_max_z)
          plane_max_z = theP.z;
        if(theP.x < plane_min_x)
          plane_min_x = theP.x;
        if(theP.y < plane_min_y)
          plane_min_y = theP.y;
        if(theP.z < plane_min_z)
          plane_min_z = theP.z;
        pcl::PointXYZRGB newP;
        newP.x = theP.x;
        newP.y = theP.y;
        newP.z = theP.z;
        newP.r = theP.r;
        newP.g = theP.g;
        newP.b = theP.b;
        RANSAC_cloud3->push_back(newP);
      }
    }
    plane_variance = pow( pow(plane_max_x - plane_min_x,2.)+pow(plane_max_y - plane_min_y,2.)+pow(plane_max_z - plane_min_z,2.) , 0.5);
    scaleAddCloud(add_cloud_ptr3,plane_variance,0.7);

    pcl::ModelCoefficients::Ptr coefficients3 (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers3 (new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZRGB> seg3;
    // Optional
    seg3.setOptimizeCoefficients (true);
    // Mandatory
    seg3.setModelType (pcl::SACMODEL_PLANE);
    seg3.setMethodType (pcl::SAC_RANSAC);
    seg3.setDistanceThreshold (ransacThreshold);
  
    seg3.setInputCloud (RANSAC_cloud3);
    seg3.segment (*inliers3, *coefficients3);
  
    if (inliers3->indices.size () == 0)
        PCL_ERROR ("Could not estimate a planar model for the given dataset.");
  
    target_nor_x = coefficients3->values[0];
    target_nor_y = coefficients3->values[1];
    target_nor_z = coefficients3->values[2];
    std::cout<<"nor: "<<target_nor_x<<" "<<target_nor_y<<" "<<target_nor_z<<endl;

    avn_x = 1;
    avn_y = 0;
    avn_z = 0;
    
    Eigen::Affine3f trans_matrix3 = Eigen::Affine3f::Identity();
    // Define a translation of 2.5 meters on the x axis.(x,y,z)
    trans_matrix3.translation() << target_pos_x - avp_x+0.06, target_pos_y - avp_y, target_pos_z - avp_z;
  
    alpha = acos( (target_nor_y*avn_y + target_nor_z*avn_z) /sqrt(target_nor_y*target_nor_y + target_nor_z*target_nor_z) );
    beta = acos( (target_nor_z*avn_z + target_nor_x*avn_x)/sqrt(target_nor_x*target_nor_x + target_nor_z*target_nor_z) );
    gamma = acos( (target_nor_y*avn_y + target_nor_x*avn_x) /sqrt(target_nor_y*target_nor_y + target_nor_x*target_nor_x) );
    //float gamma = acos( (target_nor_x*avn_x + target_nor_y*avn_y)/sqrt(target_nor_x*target_nor_x + target_nor_y*target_nor_y) );
    //std::cout<<alpha<<","<<beta<<"  ;"<<mean<<endl;
  
    trans_matrix3.rotate ( Eigen::AngleAxisf (alpha, Eigen::Vector3f::UnitX()) * Eigen::AngleAxisf (beta, Eigen::Vector3f::UnitY()) );//* Eigen::AngleAxisf (gamma, Eigen::Vector3f::UnitZ()) );
  
    // Print the transformation
    printf ("\nUsing an Affine3f derive trans_matrix:\n");
    std::cout << trans_matrix3.matrix() << std::endl;
  
    pcl::transformPointCloud(*add_cloud_ptr3, *augment_cloud3, trans_matrix3);

    //replaceRGB
    std::cerr<<"Total point:"<<augment_cloud3->points.size()<<", fitting "<<RANSAC_cloud3->points.size()<<endl;
    //replaceRGB_AR(augment_cloud3,result_cloud_ptr,AR_planar);

    for(size_t i =0; i<augment_cloud->points.size();i++){
        pcl::PointXYZRGB theP = augment_cloud->points.at(i);
        pcl::PointXYZRGB newP;
        newP.x = theP.x;
        newP.y = theP.y;
        newP.z = theP.z;
        newP.r = theP.r;
        newP.g = theP.g;
        newP.b = theP.b;
        rgb_cloud_ptr->points.push_back(newP);
      }
      for(size_t i =0; i<augment_cloud2->points.size();i++){
        pcl::PointXYZRGB theP = augment_cloud2->points.at(i);
        pcl::PointXYZRGB newP;
        newP.x = theP.x;
        newP.y = theP.y;
        newP.z = theP.z;
        newP.r = theP.r;
        newP.g = theP.g;
        newP.b = theP.b;
        rgb_cloud_ptr->points.push_back(newP);
      }
      for(size_t i =0; i<augment_cloud3->points.size();i++){
        pcl::PointXYZRGB theP = augment_cloud3->points.at(i);
        pcl::PointXYZRGB newP;
        newP.x = theP.x;
        newP.y = theP.y;
        newP.z = theP.z;
        newP.r = theP.r;
        newP.g = theP.g;
        newP.b = theP.b;
        rgb_cloud_ptr->points.push_back(newP);
      }

  }

  /// -----------------------------------|  Visualization  |-----------------------------------
  if (show_result)
  {
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addPointCloud (mpss_labeled_cloud, "maincloud");
    // uncomment below to see the original augmenting cloud!
    /*if (aug_pc){ 
      viewer->addPointCloud (augment_cloud, "augment_cloud");
      viewer->addPointCloud (augment_cloud2, "augment_cloud2");
      viewer->addPointCloud (augment_cloud3, "augment_cloud3");
    }*/
    /// Visualization Loop
    PCL_INFO ("Loading viewer\n");
    while (!viewer->wasStopped ()){
      viewer->spinOnce (100);
    }
    //self-design to show supervoxel cloud (deprecated from command line)
      if(show_svcloud){
      boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_sup (new pcl::visualization::PCLVisualizer ("3D Viewer"));
      viewer_sup->setBackgroundColor (0, 0, 0);

      PointCloudT::Ptr voxel_centroid_cloud = super.getVoxelCentroidCloud ();
      viewer_sup->addPointCloud (voxel_centroid_cloud, "voxel centroids");
      viewer_sup->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,2.0, "voxel centroids");
      viewer_sup->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY,0.95, "voxel centroids");

      PointLCloudT::Ptr labeled_voxel_cloud = super.getLabeledVoxelCloud ();
      viewer_sup->addPointCloud (labeled_voxel_cloud, "labeled voxels");
      viewer_sup->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY,0.8, "labeled voxels");
      //Normal visulizer
      //PointNCloudT::Ptr sv_normal_cloud = super.makeSupervoxelNormalCloud (supervoxel_clusters);
      //viewer_sup->addPointCloudNormals<pcl::PointNormal> (sv_normal_cloud,1,0.05f, "supervoxel_normals");

      std::multimap<uint32_t,uint32_t>::iterator label_itr = supervoxel_adjacency.begin ();
      for ( ; label_itr != supervoxel_adjacency.end (); )
      {
        //First get the label
        uint32_t supervoxel_label = label_itr->first;
        //Now get the supervoxel corresponding to the label
        pcl::Supervoxel<PointT>::Ptr supervoxel = supervoxel_clusters.at (supervoxel_label);

        //Now we need to iterate through the adjacent supervoxels and make a point cloud of them
        PointCloudT adjacent_supervoxel_centers;
        std::multimap<uint32_t,uint32_t>::iterator adjacent_itr = supervoxel_adjacency.equal_range (supervoxel_label).first;
        for ( ; adjacent_itr!=supervoxel_adjacency.equal_range (supervoxel_label).second; ++adjacent_itr)
        {
          pcl::Supervoxel<PointT>::Ptr neighbor_supervoxel = supervoxel_clusters.at (adjacent_itr->second);
          adjacent_supervoxel_centers.push_back (neighbor_supervoxel->centroid_);
        }
        //Now we make a name for this polygon
        std::stringstream ss;
        ss << "supervoxel_" << supervoxel_label;
        //This function is shown below, but is beyond the scope of this tutorial - basically it just generates a "star" polygon mesh from the points given
        addSupervoxelConnectionsToViewer (supervoxel->centroid_, adjacent_supervoxel_centers, ss.str (), viewer_sup);
        //Move iterator forward to next label
        label_itr = supervoxel_adjacency.upper_bound (supervoxel_label);
      }

      while (!viewer_sup->wasStopped ())
      {
      viewer_sup->spinOnce (100);
      }

    }
  }

  return (0);

}  /// END main



/// -------------------------| Definitions of helper functions|-------------------------

void
addSupervoxelConnectionsToViewer (PointT &supervoxel_center,
                                  PointCloudT &adjacent_supervoxel_centers,
                                  std::string supervoxel_name,
                                  boost::shared_ptr<pcl::visualization::PCLVisualizer> & viewer)
{
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New ();
  vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New ();
  vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New ();

  //Iterate through all adjacent points, and add a center point to adjacent point pair
  PointCloudT::iterator adjacent_itr = adjacent_supervoxel_centers.begin ();
  for ( ; adjacent_itr != adjacent_supervoxel_centers.end (); ++adjacent_itr)
  {
    points->InsertNextPoint (supervoxel_center.data);
    points->InsertNextPoint (adjacent_itr->data);
  }
  // Create a polydata to store everything in
  vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New ();
  // Add the points to the dataset
  polyData->SetPoints (points);
  polyLine->GetPointIds  ()->SetNumberOfIds(points->GetNumberOfPoints ());
  for(unsigned int i = 0; i < points->GetNumberOfPoints (); i++)
    polyLine->GetPointIds ()->SetId (i,i);
  cells->InsertNextCell (polyLine);
  // Add the lines to the dataset
  polyData->SetLines (cells);
  viewer->addModelFromPolyData (polyData,supervoxel_name);
}

void
savePCDfile(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const char* fileName)
{
    cloud->width = cloud->points.size();
    cloud->height = 1;
    pcl::io::savePCDFileASCII(fileName, *cloud);
}

void
findNeighbor(std::vector<uint32_t>& plane,const uint32_t& the_cluster_num, double& the_normal_x, double& the_normal_y, double& the_normal_z)
{
  plane.push_back(the_cluster_num);
  int the_cluster_int = clusters_int.find(the_cluster_num)->second;
  ::avn_x += normal_vector_x[the_cluster_int];
  ::avn_y += normal_vector_y[the_cluster_int];
  ::avn_z += normal_vector_z[the_cluster_int];
  ::avp_x += pos_x[the_cluster_int];
  ::avp_y += pos_y[the_cluster_int];
  ::avp_z += pos_z[the_cluster_int];
  ::size_temp++;
  std::multimap<uint32_t,uint32_t>::iterator adjacency_itr = supervoxel_adjacency.begin ();
  std::pair <std::multimap<uint32_t,uint32_t>::iterator, std::multimap<uint32_t,uint32_t>::iterator> range 
                                                      = supervoxel_adjacency.equal_range(the_cluster_num);
  for(adjacency_itr = range.first; adjacency_itr != range.second; adjacency_itr++){
    uint32_t neighbor_cluster = adjacency_itr->second;
    int neighbor_cluster_int = clusters_int.find(neighbor_cluster)->second;
    // Check whether the neighbor has normals like a plane
    // Supervoxel has normal of 1
    //double adj_parrallel_threshold = parrallel_threshold * (1+(pos_z[neighbor_cluster_int] - min_z));
    if(the_normal_x * normal_vector_x[neighbor_cluster_int] + the_normal_y * normal_vector_y[neighbor_cluster_int] +
        the_normal_z * normal_vector_z[neighbor_cluster_int] > parrallel_threshold && clusters_used.find(neighbor_cluster)->second == false){
      clusters_used.find(neighbor_cluster)->second = true;
      the_normal_x = (1-mu)*the_normal_x+mu*normal_vector_x[neighbor_cluster_int];
      the_normal_y = (1-mu)*the_normal_y+mu*normal_vector_y[neighbor_cluster_int];
      the_normal_z = (1-mu)*the_normal_z+mu*normal_vector_z[neighbor_cluster_int];
      findNeighbor(plane,neighbor_cluster,the_normal_x,the_normal_y,the_normal_z);
    }
  }
  return;
}

bool 
loadPointCloudFile(const std::string& fileName, pcl::PCLPointCloud2& pointCloud)
{
  if (fileName.find(".pcd") != std::string::npos) {
    printf("Load PCD file...");
    if (pcl::io::loadPCDFile(fileName.c_str(), pointCloud)) {
      printf("Fail!\n");
      return true;
    }
    printf("Success!\n");
  }
  else if (fileName.find(".ply") != std::string::npos) {
    printf("Load PLY file...");
    if (pcl::io::loadPLYFile(fileName.c_str(), pointCloud)) {
      printf("Fail!\n");
      return true;
    }
    printf("Success!\n");
  }
  else {
    printf("Not supported input format!\n");
    return true;
  }

    return false;
}


void
scaleAddCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr add_cloud_ptr, double plane_variance, double scale_ratio){
  double add_max_x = DBL_MIN;
  double add_max_y = DBL_MIN;
  double add_max_z = DBL_MIN;
  double add_min_x = DBL_MAX;
  double add_min_y = DBL_MAX;
  double add_min_z = DBL_MAX;
  for(size_t i = 0; i<add_cloud_ptr->points.size();i++){
    pcl::PointXYZRGB theP = add_cloud_ptr->points.at(i);
    if(theP.x > add_max_x)
      add_max_x = theP.x;
    if(theP.y > add_max_y)
      add_max_y = theP.y;
    if(theP.z > add_max_z)
      add_max_z = theP.z;
    if(theP.x < add_min_x)
      add_min_x = theP.x;
    if(theP.y < add_min_y)
      add_min_y = theP.y;
    if(theP.z < add_min_z)
      add_min_z = theP.z;
  }
  double add_variance = pow( pow(add_max_x-add_min_x,2.)+pow(add_max_y-add_min_y,2.)+pow(add_max_z-add_min_z,2.), 0.5 );
  double ratio = (plane_variance / add_variance) * scale_ratio;
  for(size_t i =0; i<add_cloud_ptr->points.size(); i++){
    add_cloud_ptr->points.at(i).x = add_cloud_ptr->points.at(i).x*ratio;
    add_cloud_ptr->points.at(i).y = add_cloud_ptr->points.at(i).y*ratio;
  }
}

void
replaceRGB_AR(pcl::PointCloud<pcl::PointXYZRGB>::Ptr augment_cloud, pcl::PointCloud<pcl::PointXYZRGBL>::Ptr result_cloud_ptr,size_t AR_planar){
  int count = 0;
  for(size_t i =0; i<result_cloud_ptr->points.size();i++){
    pcl::PointXYZRGBL theP = result_cloud_ptr->points.at(i);
    if(theP.label == AR_planar+2){
      if(i%1000==0)
        std::cerr<<i<<endl;
      float new_x=0;
      float new_y=0;
      float new_z=0;
      size_t j = findProjectPoint(result_cloud_ptr->points.at(i).x, result_cloud_ptr->points.at(i).y, result_cloud_ptr->points.at(i).z, augment_cloud, AR_planar+2, new_x,new_y,new_z);
      /*if(j == SIZE_MAX){
        //std::cerr<<"interpolate new point:"<< new_x<<new_y<<new_z;
        pcl::PointXYZRGBL newP;
        newP.x = new_x;
        newP.y = new_y;
        newP.z = new_z;
        newP.r = augment_cloud->points.at(i).r;
        newP.g = augment_cloud->points.at(i).g;
        newP.b = augment_cloud->points.at(i).b;
        newP.label = AR_planar+2;
        result_cloud_ptr->points.push_back(newP);
        count++;
      }*/
      if(j == SIZE_MAX)
        continue;
      else{
        result_cloud_ptr->points.at(i).r = augment_cloud->points.at(j).r;
        result_cloud_ptr->points.at(i).g = augment_cloud->points.at(j).g;
        result_cloud_ptr->points.at(i).b = augment_cloud->points.at(j).b;
      }
    }
  }
  //std::cerr<<endl<<"interpolate "<<count<<" points !!!"<<endl<<endl;
}

size_t
findProjectPoint(float x, float y, float z, pcl::PointCloud<pcl::PointXYZRGB>::Ptr augment_cloud, size_t AR_planar_label, float& new_x ,float& new_y, float& new_z)
{
  size_t nearestP = 0;
  //size_t nearestP2 = 0;
  double temp_dis = DBL_MAX;
  double min_dis = DBL_MAX;
  //double min_dis2 = DBL_MAX;
  for(size_t i =0; i<augment_cloud->points.size();i++){
    pcl::PointXYZRGB theP = augment_cloud->points.at(i);
    temp_dis = std::sqrt( std::pow(x - augment_cloud->points.at(i).x,2)+std::pow(y - augment_cloud->points.at(i).y,2)+std::pow(z - augment_cloud->points.at(i).z,2) );
    if( temp_dis < min_dis){
      min_dis = temp_dis;
      nearestP = i;
    }
    /*if( temp_dis < min_dis){
      min_dis2 = min_dis;
      min_dis = temp_dis;
      nearestP2 = nearestP;
      nearestP = i;
    }
    else if(temp_dis < min_dis2){
      min_dis2 = temp_dis;
      nearestP2 = i;
    }*/
  }
  //std::cerr<<min_dis<<endl;
  /*if(min_dis > 0.04){
    pcl::PointXYZRGBL nearP = result_cloud_ptr->points.at(nearestP);
    pcl::PointXYZRGBL nearP2 = result_cloud_ptr->points.at(nearestP2);
    new_x = (nearP.x + nearP2.x)/2.;
    new_y = (nearP.y + nearP2.y)/2.;
    new_z = (nearP.z + nearP2.z)/2.;
    return SIZE_MAX;
  }
  else*/
  if(min_dis > 0.1)
    return SIZE_MAX;
  return nearestP;
}
