#include <cstdlib>
#include <cmath>
#include <climits>
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

//time usage
#include <ctime>
using namespace std;

bool loadPointCloudFile(const string& fileName, pcl::PCLPointCloud2& pointCloud);
size_t findLabel(uint32_t la, pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud_ptr);

int main(int argc, char ** argv){
	if (argc == 1) {
    PCL_INFO("Usage: ./newPlanarRefinements [old_pc] [old_info] [2add_pc] [2add_info] [output_pc] [output_info] (-sr)\n");
    PCL_INFO("  Ex:  ./newPlanarRefinements 30_xyzrgbl.ply 30_POS_NORM_VARs.txt 40_xyzrgbl.ply 40_POS_NORM_VARs.txt 30+40_xyzrgbl.ply 30+40_POS_NORM_VARs.txt -sr\n");
    PCL_INFO("Notice:\n");
    PCL_INFO("  [old_pc],[2add_pc] supports only  the output of MPSS (format: pcl::PointXYZRGB)\n");
    PCL_INFO("  [old_info] needs to be align to same file as [old_pc]\n");
    PCL_INFO("  [2add_info] also needs to be align to same file as [2add_pc]\n");
    PCL_INFO("  -sr: show result\n");
    return false;
  }
  bool has_normals = false;

  float parrallel_filter = 0.8;
  float distance_to_plane = 0.002;
  float curve_threshold = 0.1;

  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr input_cloud_ptr1 (new pcl::PointCloud<pcl::PointXYZRGBL>);
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr input_cloud_ptr2 (new pcl::PointCloud<pcl::PointXYZRGBL>);

  bool show_result = pcl::console::find_switch (argc, argv, "-sr");
  string save_plyname = argv[5];
  string save_txtname = argv[6];
  // ================= Read first file =======================
  /// Get pcd path from command line
  string pcd_filename1 = argv[1];
  PCL_INFO ("Loading pointcloud\n");
  
  /// check if the provided pcd file contains normals
  pcl::PCLPointCloud2 input_pointcloud1;  //inpu_pointcloud2 ,new version of pcl
  if (loadPointCloudFile(pcd_filename1, input_pointcloud1))
  {
    PCL_ERROR ("ERROR: Could not read input point cloud %s.\n", pcd_filename1.c_str ());
    return (3);
  }
  pcl::fromPCLPointCloud2 (input_pointcloud1, *input_cloud_ptr1);

  string txt_filename1 = argv[2];
  ifstream txt1(txt_filename1);
  string line;
  string buf;
  vector< vector<float> > txtVectors1;
  vector<float> txtvector1;
  while( getline(txt1,line) ){
    txtvector1.clear();
    stringstream ss(line);
    while(ss >> buf){
      txtvector1.push_back( stof(buf) );
    }
    txtVectors1.push_back(txtvector1);
  }
  txt1.close();
  if(input_cloud_ptr1->size() != txtVectors1.size()){
    PCL_ERROR ("ERROR: SIZE MISMATCH! \n");
    PCL_ERROR(" '%s' has %d points, while '%s' has %d points.\n",pcd_filename1.c_str (),input_cloud_ptr1->size(),txt_filename1.c_str(),txtVectors1.size());
    return (3);
  }
  else
    PCL_INFO("Done reading cloud & info\n");

  // ================= Read second file ====================
  string pcd_filename2 = argv[3];
  PCL_INFO ("Loading pointcloud\n");
  
  /// check if the provided pcd file contains normals
  pcl::PCLPointCloud2 input_pointcloud2;  //inpu_pointcloud2 ,new version of pcl
  if (loadPointCloudFile(pcd_filename2, input_pointcloud2))
  {
    PCL_ERROR ("ERROR: Could not read input point cloud %s.\n", pcd_filename2.c_str ());
    return (3);
  }
  pcl::fromPCLPointCloud2 (input_pointcloud2, *input_cloud_ptr2);

  string txt_filename2 = argv[4];
  ifstream txt2(txt_filename2);
  vector< vector<float> > txtVectors2;
  vector<float> txtvector2;
  line.clear();
  buf.clear();
  while( getline(txt2,line) ){
    txtvector2.clear();
    stringstream ss(line);
    while(ss >> buf){
      txtvector2.push_back( stof(buf) );
    }
    txtVectors2.push_back(txtvector2);
  }
  txt2.close();
  if(input_cloud_ptr2->size() != txtVectors2.size()){
    PCL_ERROR ("ERROR: SIZE MISMATCH! \n");
    PCL_ERROR (" '%s' has %d points, while '%s' has %d points.\n",pcd_filename2.c_str (),input_cloud_ptr2->size(),txt_filename2.c_str(),txtVectors2.size());
    return (3);
  }
  else
    PCL_INFO("Done reading cloud & info\n");
  //cout<<txtVectors2.size()<<endl;
  // ================ processing ==================
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr result_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGBL>);
  result_cloud_ptr = input_cloud_ptr1;
  vector< vector<float> > resultVectors = txtVectors1;
  uint32_t max_label = 0 ;
  uint32_t max2add_label = 0;
  for(size_t i=0; i != result_cloud_ptr->points.size();i++){
    pcl::PointXYZRGBL inputP = result_cloud_ptr->points.at(i);
    if(inputP.label > max_label)
      max_label = inputP.label;
  }
  for(size_t i =0; i!= input_cloud_ptr2->points.size();i++){
    pcl::PointXYZRGBL addP = input_cloud_ptr2->points.at(i);
    if(addP.label > max2add_label)
      max2add_label = addP.label;
  }
  float pos_x_2add,pos_y_2add,pos_z_2add,norm_x_2add,norm_y_2add,norm_z_2add;
  float pos_x,pos_y,pos_z,norm_x,norm_y,norm_z;
  uint32_t next_label = max_label+1;
  cout<<max_label<<","<<max2add_label<<endl;
  for(uint32_t l=2; l <= max2add_label;l++){
    cerr<<l<<endl;
    // found a representitive of a specific label "l"
    size_t pivot = findLabel(l,input_cloud_ptr2);
    if(pivot == -1)
      continue;
    if(txtVectors2[pivot][6] > curve_threshold )
      continue;
    pos_x_2add = txtVectors2[pivot][0];
    pos_y_2add = txtVectors2[pivot][1];
    pos_z_2add = txtVectors2[pivot][2];
    norm_x_2add = txtVectors2[pivot][3];
    norm_y_2add = txtVectors2[pivot][4];
    norm_z_2add = txtVectors2[pivot][5];
    bool planar_combined = false;
    cerr<<"...Checking original cloud (may take a few seconds)"<<endl;
    for(uint32_t la=2 ; la <= max_label; la++){
      size_t hole = findLabel(la,input_cloud_ptr1);
      if(hole == -1)
        continue;
      if(txtVectors1[hole][6] > curve_threshold)
        continue;
      pos_x = txtVectors1[hole][0];
      pos_y = txtVectors1[hole][1];
      pos_z = txtVectors1[hole][2];
      norm_x = txtVectors1[hole][3];
      norm_y = txtVectors1[hole][4];
      norm_z = txtVectors1[hole][5];
      if( abs(norm_x_2add*norm_x+norm_y_2add*norm_y+norm_z_2add*norm_z) > parrallel_filter && 
          abs((norm_x_2add*pos_x_2add + norm_y_2add*pos_y_2add + norm_z_2add*pos_z_2add) - (norm_x*pos_x + norm_y*pos_y + norm_z*pos_z)) < distance_to_plane ){
        //planar refinements
        bool planar_combined = true;
        cerr<<"......Combining planes (may take a few seconds)"<<endl;
        for(size_t i = 0;i<input_cloud_ptr2->size();i++){
          if(input_cloud_ptr2->points.at(i).label == l){
            pcl::PointXYZRGBL newP = input_cloud_ptr2->points.at(i);
            newP.label = la;
            result_cloud_ptr->push_back(newP);
            resultVectors.push_back(txtVectors2[i]);
          }
        }
      }
    }
    if(planar_combined == false){
      for(size_t i = 0;i<input_cloud_ptr2->size();i++){
        if(input_cloud_ptr2->points.at(i).label == l){
          pcl::PointXYZRGBL newP = input_cloud_ptr2->points.at(i);
          newP.label = next_label;
          result_cloud_ptr->push_back(newP);
          resultVectors.push_back(txtVectors2[i]);
        }
      }
      next_label++;
    }
  }

  pcl::io::savePLYFileASCII(save_plyname, *result_cloud_ptr);
  ofstream oFile;
  oFile.open(save_txtname);
  for(size_t i = 0; i<resultVectors.size();i++){
    for(size_t j =0; j<resultVectors[i].size();j++){
      oFile << resultVectors[i][j] << " ";
    }
    oFile <<"\n";
  }
  oFile.close();

  if(show_result){
    pcl::PointCloud<pcl::PointXYZL>::Ptr show_result_ptr (new pcl::PointCloud<pcl::PointXYZL>);
    for(size_t i = 0; i <result_cloud_ptr->points.size();i++){
      pcl::PointXYZL showP;
      pcl::PointXYZRGBL resultP = result_cloud_ptr->points.at(i);
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

	return (0);
}

size_t
findLabel(uint32_t la, pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud_ptr)
{
  size_t pivot = 0;
  bool cantFind = true;
  for(;pivot != cloud_ptr->points.size();pivot++){
    if(cloud_ptr->points.at(pivot).label == la){
      cantFind = false;
      break;
    }
  }
  if(cantFind == true)
    return -1;
  return pivot;
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