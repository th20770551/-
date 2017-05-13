#include<iostream>
#include<fstream>
#include<string>
#include<stdlib.h>
#include<vector>
#include<map>
#include<math.h>
#include<float.h>

#include<sys/types.h>//ディレクトリ内の全ファイルの名前を取得するために使用
#include<dirent.h>

using namespace std;

#define EPSILON 0.000001 


//指定されたディレクトリ中に存在するファイル名の一覧を取得
void get_filename_list(string filename , vector<string> &name_list){
	
	//ファイル名の一覧を取得
	DIR* data_dir = opendir( filename.c_str() );
	if(data_dir != NULL){
		struct dirent* dent;
		do{
			dent = readdir(data_dir);
			if(dent != NULL){
				//cout << dent->d_name << endl;
				string temp = dent->d_name;
				if(temp[0] != '.')//先頭が.はファイル名ではない
					name_list.push_back(temp);
			}
		}while(dent != NULL);
		closedir(data_dir);
	}	
}


//指定された名前のディレクトリ中のファイルから、特徴を読み込み
void get_bf_feature( vector<string> &name_list , vector< vector<float> > &target_data , int dimension_of_feature , string root_dir){

	for(int i = 0 ; i < name_list.size() ; i++){
		target_data[i].resize(dimension_of_feature);
		string reading_filename = root_dir + '/' + name_list[i];
		//cout << reading_filename << endl;
		ifstream ifs(reading_filename.c_str());
		for(int j = 0 ; j < dimension_of_feature ; j++){
			float temp;
			ifs >> temp;
			//cout << temp << " ";
			target_data[i][j] = temp;
		}
		//cout << endl;
		ifs.close();
	}
}

	
void calc_distance( vector<string> &query_name_list , vector< vector<float> > &query_data 
					, vector<string> &target_name_list , vector< vector<float> > &target_data
					,string distance , string rlt_output_dirname , int dimension_of_feature){
	
	//rltファイルを保存するディレクトリを作成
	string LINUX_COMMAND = "mkdir " + rlt_output_dirname;
    int RUN = system(LINUX_COMMAND.c_str());	
	
	//クエリファイルごとに処理していく
	for(int i = 0 ; i < query_name_list.size() ; i++){
		
		//rltファイルに書き出す距離　mapクラスを使って自動的にソートする
		map< float , string > rlt_distance;
		
		for(int j = 0 ; j < target_name_list.size() ; j++){
			
			float temp_dist = 0.0f;
			
			if(distance == "L1"){	
				for(int k = 0 ; k < dimension_of_feature ; k++){
					temp_dist += fabsf( query_data[i][k] - target_data[j][k] );
				}
			}else if(distance == "L2"){
				for(int k = 0 ; k < dimension_of_feature ; k++){
					temp_dist += pow( query_data[i][k] - target_data[j][k] ,2);
				}				
				temp_dist = sqrtf(temp_dist);
			}else if(distance == "KLD"){
				
				for(int k = 0 ; k < dimension_of_feature ; k++){
					float p = query_data[i][k];
					float q = target_data[j][k];
					
					if(fabs(p) < EPSILON)
						p = EPSILON;
					
					if(fabs(q) < EPSILON)
						q = EPSILON;
					
					temp_dist += (q - p) * logf(q / p);
				}
			}
			if(rlt_distance[temp_dist] != ""){
				cout << "同着距離の被り発見　便宜上、微妙に距離をズラして同着問題を解消\n";
				while(rlt_distance[temp_dist] != "")
					temp_dist += EPSILON;
			}
			
			rlt_distance[temp_dist] = target_name_list[j].substr(0,7) + target_name_list[j].substr(8,4);
			
		}
		
		string output_rlt_filename = rlt_output_dirname + '/' + query_name_list[i].substr(0,7) + query_name_list[i].substr(8,4) + ".rlt";
		ofstream ofs(output_rlt_filename.c_str());
		ofs << target_name_list.size() << endl;
		
		for(map<float,string>::iterator itr = rlt_distance.begin(); itr != rlt_distance.end() ; itr++){
			ofs << itr->second << ' ' << itr->first << endl;
		}
		
		ofs.close();
	}
	
}


int main(int argc, char* argv[]){
	/*
		コマンドライン引数
			＜検索対象ディレクトリ名＞　＜検索クエリディレクトリ名＞　＜１つの統合特徴の次元数＞　＜距離尺度＞ <rltの書き出しディレクトリ名>
	*/
	if(argc != 6){
		cout << "Usage : " << argv[0] << " <target dir> <query dir> <dimension of feature> <distance> <rlt output dirname>\n";
		return 0;
	}
	
	string target_dir = argv[1];
	string query_dir  = argv[2];
	int dimension_of_feature = atoi(argv[3]);
	string distance = argv[4];
	string rlt_output_dirname = argv[5];
	
	cout << "検索ターゲットディレクトリ : " << target_dir << endl;
	cout << "検索クエリディレクトリ : " << query_dir << endl;
	cout << "統合特徴の次元数 : " << dimension_of_feature << endl;
	cout << "使用する距離尺度 : " << distance << endl;
	cout << "rlt書き出し先ディレクトリ名 : " << rlt_output_dirname << endl;
	
	
	if( !(distance == "L1" || distance == "L2" || distance == "KLD") ){
		cout << "cannot detect " << distance << " distance\n";
		return 0;
	}
	
	//検索対象とクエリのファイルの名前一覧
	vector<string> target_name_list;
	vector<string> query_name_list;
	
	get_filename_list(target_dir,target_name_list);
	get_filename_list(query_dir,query_name_list);
	
	vector< vector<float> > target_data , query_data;
	target_data.resize(target_name_list.size());
	query_data.resize(query_name_list.size());
	
	
	get_bf_feature( target_name_list , target_data , dimension_of_feature ,target_dir);
	get_bf_feature( query_name_list , query_data , dimension_of_feature , query_dir);
	
	
	//引数　検索クエリ名　、　検索クエリデータ　、　検索ターゲット名　、　検索ターゲットデータ　、　距離尺度　、　rltの書き出し先ディレクトリ
	calc_distance( query_name_list , query_data , target_name_list , target_data , distance , rlt_output_dirname , dimension_of_feature );
	
	
	return 0;
}
