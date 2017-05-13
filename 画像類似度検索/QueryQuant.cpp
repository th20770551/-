/*
	検索クエリ画像の量子化プログラム
*/

#include<iostream>
#include<string>
#include<stdlib.h>
#include<fstream>
#include<vector>

#include<sys/types.h>//ディレクトリ内の全ファイルの名前を取得するために使用
#include<dirent.h>
#include<math.h>

using namespace std;

int main(int argc , char* argv[]){

	/*
	やっている事
		1.クラスタ重心の.txtファイルを読み込む
		2.クエリ画像のSIFT特徴量を読み込む
		3.CVkmeansと同じように、ベクトル量子化する
		4.統合特徴量を、「output_bf_query_(単語数)」ディレクトリ内に保存する
	*/

	//コマンドライン引数で、クラスタファイル名　クラスタ数　クエリSIFT特徴のディレクトリを受け取る
	
	if(argc != 4){
		cout << "usage : " << argv[0] << " <cluster_file.txt>  <cluster_num>  <query_sift_dir>\n";
		return 0;
	}
	
	string cluster_filename = argv[1];
	int cluster_num = atoi(argv[2]);
	string query_sift_dirname = argv[3];
	
	cout << "クラスタファイル : " << cluster_filename << endl;
	cout << "クラスタ数 : " << cluster_num << endl;
	cout << "クエリディレクトリ : " << query_sift_dirname << endl;
	
	
	//クラスタ重心の読み込み
	float centers[cluster_num][128];//クラスタ重心の格納用
	ifstream ifs(cluster_filename.c_str());	
	for(int y = 0 ; y < cluster_num ; y++){
		for(int x = 0 ; x < 128 ; x++){
				float temp;
				ifs >> temp;
				centers[y][x] = temp;
		}
	}
	ifs.close();
	
	
	//SIFT特徴を記録したテキストファイルの名前一覧を取得
	vector<string> sift_text_name;//SIFT特徴のテキストファイル名配列
	DIR* sift_dir = opendir( query_sift_dirname.c_str() );
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
		
	/*
	for(int i = 0 ; i < sift_text_name.size() ; i++){
		cout << sift_text_name[i] << endl;
	}
	cout << "size : " << sift_text_name.size() << endl;*/
	

	//統合特徴の書き出し先ディレクトリ
	string bf_feature_output_dir = query_sift_dirname + "_bf_" + to_string(cluster_num);
	string command = "mkdir -p " + bf_feature_output_dir;
	system( command.c_str() );

	
	//クエリのSIFT特徴量を読み込んで、量子化。そのまま書き出しをループで繰り返す
	for(int i = 0 ; i < sift_text_name.size(); i++){
		string query_sift_filename = query_sift_dirname + '/' + sift_text_name[i];
		
		//cout << query_sift_filename << endl;
		
		ifstream ifs_sift(query_sift_filename.c_str());
		
		//行数をカウント
		int line_count = 0;
		int count = 0;
		float temp;
		while(!ifs_sift.eof()){
			ifs_sift >> temp;
			float temp2 = temp;
			count++;
			if(count > 127){
				line_count++;
				count = 0;
			}
		}
		//cout << line_count << endl;
		
		
		ifs_sift.close();
		
		//現在のループで読むファイルに格納されているsiftを保存するための配列
		float this_sift_feature[line_count][128];
		
		float bf_feature[cluster_num];//ベクトル量子化済みの特徴を格納する配列
		for(int i = 0 ; i < cluster_num ; i++){
			bf_feature[i] = 0.0f;
		}
		
		//情報を読み込み
		ifstream ifs_sift2(query_sift_filename.c_str());		
		for(int y = 0 ; y < line_count ; y++){
			for(int x = 0 ; x < 128 ; x++){
					ifs_sift2 >> temp;
					this_sift_feature[y][x] = temp;
			}
		}
		ifs_sift2.close();
		
		

		for(int j = 0 ; j < line_count ; ++j){
			//各行のSIFT特徴を読み込み
			float this_sift[128];
			for(int k = 0 ; k < 128 ; k++){
				this_sift[k] = this_sift_feature[j][k];
			}			
			
			//各行ごとに、重心と距離計算
			float min_distance = -1;
			int min_distance_cluster_no = -1;
			
			for(int l = 0 ; l < cluster_num ; ++l){
				float distance = 0.0f;
				for(int m = 0 ; m < 128 ; ++m){
					distance += pow(this_sift[m] - centers[l][m],2);
				}
				distance = sqrt(distance);
				
				if(min_distance < 0 || min_distance > distance){
					min_distance = distance;
					min_distance_cluster_no = l;
				}
			}
			bf_feature[min_distance_cluster_no] += 1;			
		}

		//統合特徴をL1正規化
		//統合特徴の書き出し　書き出しの際に、L1正規化するbf_feature_output_dir
		string output_bf_filename;
		output_bf_filename = bf_feature_output_dir + '/' + sift_text_name[i] + "_bf.txt";
		
		ofstream ofs_bf(output_bf_filename.c_str());
		for(int n = 0 ; n < cluster_num ; n++){
			ofs_bf << (float)bf_feature[n] / (float)line_count << " ";
		}
		ofs_bf.close();		
		
		for(int ii = 0 ; ii < cluster_num ; ii++){
			bf_feature[ii] = 0.0f;
		}		

	}
	
	
	return 0;
}
