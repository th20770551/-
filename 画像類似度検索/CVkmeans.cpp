/*
	SIFT特徴のtxtファイルを読み込み、k-meansを実行。
	クラスタ重心を求める。
	クラスタ重心を求めた後は、重心をテキストファイルとして書き出し。
	クラスタ重心は、一つのテキストファイルへ保存する
*/

#include<iostream>
#include<fstream>
#include<vector>

#include<time.h>
#include <sys/times.h>//マルチコアCPU向け時間計測
#include <sys/time.h>
#include <unistd.h>

#include<math.h>

#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<sys/types.h>//ディレクトリ内の全ファイルの名前を取得するために使用
#include<dirent.h>

using namespace std;
using namespace cv;

clock_t times_clock()
{
    struct tms t;
    return times(&t);
}

int main(int argc , char *argv[]){
	
	//コマンドライン引数で、SIFT特徴の存在するディレクトリ名を取得する。
	if(argc != 3){
		cout << "usage : " << argv[0] << " <sift feature dir> <cluster num>\n";
		return -1;
	}
	
	/* SIFT特徴のファイル名取得 */
	
	//SIFT特徴の総数を取得　Matのサイズ設定に必要
	ifstream res_ifs("result_dense_64x48.txt");//SIFTextractorの処理結果メモを使用する
	if(res_ifs.fail()){
		cerr << "result.txt is not found\n";
		return -1;
	}
	string trush1 , trush2;//ゴミ捨て用
	int temp_num;//数値取得用
	int sift_amount = 0;//sift特徴の総数
	while(res_ifs >> trush1 >> trush2 >> temp_num){
		sift_amount = max(temp_num,sift_amount);
	}
	res_ifs.close();
	
	cout << "SIFT特徴量総数 : " << sift_amount << endl;
	
	//SIFT特徴を記録したテキストファイルの名前一覧を取得
	vector<string> sift_text_name;//SIFT特徴のテキストファイル名配列
	DIR* sift_dir = opendir( argv[1] );
	if(sift_dir != NULL){
		struct dirent* dent;
		do{
			dent = readdir(sift_dir);
			if(dent != NULL){
				//cout << dent->d_name << endl;
				string temp = dent->d_name;
				if(temp[0] != '.')//先頭が.はファイル名ではないので除外
					sift_text_name.push_back(temp);
			}
		}while(dent != NULL);
		closedir(sift_dir);
	}	
	
	
	/* SIFT特徴の読み込み */
	
	//各画像ごとのsift特徴の数
	int sift_feature_num[sift_text_name.size()];
	
	//全てのSIFT特徴を格納するための配列を用意 opencvのkmeansはfloat型のMat値を使うので、CV_32FC1で宣言
	Mat sift_features(Size(128,sift_amount),CV_32FC1);
	int sift_counter = 0;//Matに格納済みのsift特徴カウンター
	//sift特徴のファイルを1つ1つ読んで、Mat配列に格納していく
	cout << "SIFT特徴読み込み開始\n";
	for(int i = 0 ; i < sift_text_name.size() ; ++i){
		string filename = (string)argv[1] + '/' + sift_text_name[i];		
		ifstream ifs(filename.c_str());
		//空行を読み込まずに、sift特徴をMatへ格納する
		float temp = 0;
		int sf_count = 0;
		int sift_counter_temp = 0;//現在のファイルのsift特徴量数
		while(ifs >> temp){
			sift_features.at<float>(sift_counter,sf_count) = temp;
			sf_count++;
			if(sf_count == 128){
				sf_count = 0;
				sift_counter++;
				sift_counter_temp++;
			}
		}
		sift_feature_num[i] = sift_counter_temp;
		ifs.close();
		//cout<< sift_text_name[i] << " : " << sift_counter_temp << endl;
	}
	cout << "SIFT特徴読み込み完了\n";

	
	// Kmeansを実行 
	
	cout << "kmeans実行開始\n";
	Mat labels, centers;//labels : ラベル , centers : クラスタ重心
	//int cluster_num = 5;//000;//クラスタ数
	int cluster_num = atoi(argv[2]);
	cout << "重心数 : " << cluster_num << endl;
	TermCriteria criteria(TermCriteria::COUNT,300,1);//Kmenasの反復回数を指定 300回とする
	clock_t t1, t2;//時間計測
	
	t1 = times_clock();
	// kmeans実行 ランダム初期値 
	//kmeans(sift_features,cluster_num,labels,criteria,1,KMEANS_RANDOM_CENTERS,centers);
	// kmeans実行 PP初期値 
	kmeans(sift_features,cluster_num,labels,criteria,1,KMEANS_PP_CENTERS,centers);
	t2 = times_clock();
	
	cout << "kmeans実行完了\n";
	cout<<"所要時間 : " << (double)(t2 - t1) / sysconf(_SC_CLK_TCK) << " 秒" <<endl;
	
	
	//課題4 重心データの書き出し
	//扱いやすさを考えて、1つのテキストファイルに書き出すこととする。
	string output_cls_name;
	output_cls_name = "cluster_64x48_" + to_string(cluster_num) + ".txt";
	
	ofstream ofs(output_cls_name.c_str());
	for(int y = 0 ; y < centers.rows ; ++y){
		for(int x = 0 ; x < centers.cols ; ++x){
			ofs << centers.at<float>(y,x) << " ";
		}
		ofs << endl;
	}
	
	

	//課題4 ベクトル量子化 重心への距離尺度はL1
	
	//統合特徴の書き出し先ディレクトリ
	string bf_feature_output_dir = (string)argv[1] + "_bf_" + to_string(cluster_num);
	string command = "mkdir -p " + bf_feature_output_dir;
	system( command.c_str() );
	
	float bf_feature[cluster_num];//ベクトル量子化済みの特徴を格納する配列
	for(int i = 0 ; i < cluster_num ; i++){
		bf_feature[i] = 0.0f;
	}
	
	cout << "ベクトル量子化中" << endl;
	int line_count = 0;//全てのSIFT特徴を纏めたMATの、何行目まで読んだかを記録するカウンター
	for(int i = 0 ; i < sift_text_name.size() ; ++i){
		
		//sift_features

		for(int j = 0 ; j < sift_feature_num[i] ; ++j ){
			
			//各行のSIFT特徴を読み込み
			float this_sift[128];
			for(int k = 0 ; k < 128 ; k++){
				this_sift[k] = sift_features.at<float>(line_count+j,k);
			}
			
			//各行ごとに、重心と距離計算
			float min_distance = -1;
			int min_distance_cluster_no = -1;
			
			for(int l = 0 ; l < cluster_num ; ++l){
				
				float distance = 0;
				for(int m = 0 ; m < 128 ; ++m){
					distance += pow(this_sift[m] - centers.at<float>(l,m),2);
				}
				distance = sqrt(distance);
				
				if(min_distance < 0 || min_distance > distance){
					min_distance = distance;
					min_distance_cluster_no = l;
				}
			}
			//最も距離の違い行の番号に+1して投票する
			bf_feature[min_distance_cluster_no] += 1;
			
		}
		line_count += sift_feature_num[i];
		
		//統合特徴をL1正規化
		//統合特徴の書き出し 書き出しの際に、L1正規化をする
		string output_bf_filename; 
		output_bf_filename = bf_feature_output_dir + '/' + sift_text_name[i] + "_bf.txt";
		ofstream ofs_bf(output_bf_filename.c_str());
		for(int n = 0 ; n < cluster_num ; n++){
			ofs_bf << (float)bf_feature[n] / (float)sift_feature_num[i] << " ";
		}
		ofs_bf.close();
		
		
		for(int ii = 0 ; ii < cluster_num ; ii++){
			bf_feature[ii] = 0.0f;
		}
	}
	cout << "ベクトル量子化終了\n";

	return 0;
}
//マルチコアCPU向け時間計測参考 : http://handasse.blogspot.com/2007/06/c_26.html