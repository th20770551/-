/*
	画像データベースから読み込んだ画像を明度画像に変換し、
	明度値からSIFT特徴を抽出する。
	抽出したSIFT特徴をテキスト形式で保存するプログラム
*/

#include<cv.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/nonfree/nonfree.hpp> // SIFT用

#include<iostream>
#include<vector>
#include<string>
#include<fstream>

#include<sys/types.h>//ディレクトリ内の全ファイルの名前を取得するために使用
#include<dirent.h>

using namespace std;
using namespace cv;

int main(int argc , char *argv[]){

	//コマンドライン引数で、読み込む画像が格納されたDBパスを取得する
	if(argc != 2){
		//データベースのデータディレクトリを指定するようにする。
		cerr << "usage : " << argv[0] << " <database_root>\n";
		return -1;
	}
	
	//ファイル名の一覧を取得
	vector<string> image_name;//画像データのファイル名配列
	string data_dirname = (string)argv[1] + "/database";
	DIR* data_dir = opendir( data_dirname.c_str() );
	if(data_dir != NULL){
		struct dirent* dent;
		do{
			dent = readdir(data_dir);
			if(dent != NULL){
				//cout << dent->d_name << endl;
				string temp = dent->d_name;
				if(temp[0] != '.')//先頭が.はファイル名ではない
					image_name.push_back(temp);
			}
		}while(dent != NULL);
		closedir(data_dir);
	}

	//特徴量を保存するディレクトリを作成
	string LINUX_COMMAND = "mkdir output";
    int RUN = system(LINUX_COMMAND.c_str());

	//SIFTモジュールを使用するための宣言
	initModule_nonfree();
	
	//画像からSIFT特徴を抽出して、テキスト形式で書き出す
	unsigned int sift_amount = 0;//全ての画像に存在するSIFT特徴の総数　後で1枚あたりのSIFT特徴の数を計算するのに使う。
	for(int i = 0 ; i < image_name.size() ; i++){
		//処理する画像ファイルまでのパス
		string image_path = data_dirname + '/' + image_name[i];
		
		//特徴を書き出すテキストファイルのパス
		string output_path = "output/" + image_name[i] + ".txt";

		Mat gbr_image , lab_image;
		//gbr_image : 元のGBR画像 opencvではGBRの順で色が並ぶため
		//lab_image : L*a*b色空間に変換した画像
		
		//GBR画像の読み込み
		gbr_image = imread(image_path.c_str());

		if(gbr_image.empty()){
			cout << image_name[i] << " image is not found.\n";
			return -1;
		}
		//GBRカラーのjpg画像を、L*a*b色空間の画像へ変換
		cvtColor(gbr_image,lab_image,CV_BGR2Lab);

		//Lab画像のチャンネルを分割する。
		//SIFTを適用するためのLの値だけの画像を作る
		vector<Mat> Lab_channel;
		split(lab_image,Lab_channel);

		//SIFTのキーポイント検出器を宣言
		SiftFeatureDetector SIFTdetector;

		//キーポイントの格納配列
		vector<KeyPoint> SIFTkeypoints;

		//Lab画像のLの値に対してSIFTを抽出する
		SIFTdetector.detect(Lab_channel[0], SIFTkeypoints);

		//SIFT Extractor
		SiftDescriptorExtractor extractor;
		
		Mat sift_output;
		extractor.compute(Lab_channel[0],SIFTkeypoints,sift_output);

		//SIFT特徴量をテキスト形式で出力
		ofstream ofs(output_path.c_str());
		for(int y = 0 ; y < sift_output.rows ; ++y){
			for(int x = 0 ; x < sift_output.cols ; ++x){
				ofs << sift_output.at<float>(y,x) << " ";
			}
			if(y != sift_output.rows - 1)//空行防止
				ofs << endl;
			
			sift_amount++;
		}
		ofs.close();
	}
	
	//siftの総数などをメモとして保存 kmeansプログラムで使用する
	ofstream result_ofs("result.txt");
	result_ofs << "画像枚数 :			" << image_name.size() << endl;
	result_ofs << "SIFT総数 :		" << sift_amount << endl;
	result_ofs << "1枚あたりのSIFT特徴数 :	" << sift_amount / image_name.size() << endl;
	result_ofs << (string)argv[1] << " : 0" << endl;//データディレクトリ名をKmeansに渡す用
	result_ofs.close();
	
	return 0;
}
