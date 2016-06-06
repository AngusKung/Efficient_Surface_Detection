#include <cstdlib>
#include <climits>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>

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

using namespace std;

bool loadPointCloudFile(const string& fileName, pcl::PCLPointCloud2& pointCloud);

int main(int argc, char ** argv){
	if (argc == 1) {
    PCL_INFO("Usage: ./newPlanarRefinements [input_point_cloud]\n");
    PCL_INFO("  Ex:  ./newPlanarRefinements result/30_xyzrgbl.ply\n");
    PCL_INFO("Notice:\n");
    PCL_INFO("  [input_point_cloud] supports only  the output of MPSS (format: pcl::PointXYZRGBL)\n");
    return false;
  }

  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr input_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGBL>);
  string pcd_filename = argv[1];
  PCL_INFO ("Loading pointcloud\n");
  
  /// check if the provided pcd file contains normals
  pcl::PCLPointCloud2 input_pointcloud;  //inpu_pointcloud2 ,new version of pcl
  if (loadPointCloudFile(pcd_filename, input_pointcloud))
  {
    PCL_ERROR ("ERROR: Could not read input point cloud %s.\n", pcd_filename.c_str ());
    return (3);
  }
  pcl::fromPCLPointCloud2 (input_pointcloud, *input_cloud_ptr);
  PCL_INFO ("Done making cloud\n");

  /*pcl::PointCloud<pcl::PointXYZRGB>::Ptr show_result_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
  for(size_t i = 0; i <input_cloud_ptr->points.size();i++){
    pcl::PointXYZRGB showP;
    pcl::PointXYZRGBL resultP = input_cloud_ptr->points.at(i);
    showP.x = resultP.x;
    showP.y = resultP.y;
    showP.z = resultP.z;
    int color = 255 - 2*resultP.label;
    if(color < 60)
      color = 60;
    showP.r = color;
    showP.g = color;
    showP.b = color;
    show_result_ptr->push_back(showP);
  }*/
  pcl::PointCloud<pcl::PointXYZL>::Ptr show_result_ptr (new pcl::PointCloud<pcl::PointXYZL>);
  for(size_t i = 0; i <input_cloud_ptr->points.size();i++){
    pcl::PointXYZL showP;
    pcl::PointXYZRGBL resultP = input_cloud_ptr->points.at(i);
    showP.x = resultP.x;
    showP.y = resultP.y;
    showP.z = resultP.z;
    showP.label = resultP.label;
    show_result_ptr->push_back(showP);
  }
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud (show_result_ptr, "maincloud");
  /// Visualization Loop
  PCL_INFO ("Loading viewer\n");
  while (!viewer->wasStopped ()){
    viewer->spinOnce (100);
  }
}

bool 
loadPointCloudFile(const string& fileName, pcl::PCLPointCloud2& pointCloud)
{
  if (fileName.find(".pcd") != string::npos) {
    printf("Load PCD file...");
    if (pcl::io::loadPCDFile(fileName.c_str(), pointCloud)) {
      printf("Fail!\n");
      return true;
    }
    printf("Success!\n");
  }
  else if (fileName.find(".ply") != string::npos) {
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