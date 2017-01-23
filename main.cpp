//
// Light crayon with VLC ID and wifi Data translation
// 
//
//  Blue  Green  Red   ID
//   OFF   OFF   ON    0
//   OFF   ON    OFF   1
//   OFF   ON    ON    2
//   ON    OFF   OFF   3
//   ON    OFF   ON    4
//   ON    ON    OFF   5
//   ON    ON    ON    6
// max 7 users
//
// ipacket 6 bit
// 3 bit ID / 3bit col (joystick orientation)

//  col  joystick orientation
//  0     neutral
//  1     up
//  2     down
//  3     left
//  4     right

// ���N�����������M����f�[�^�Ƀp���e�B�r�b�g��"����Ȃ�"

#define _USE_MATH_DEFINES

#include <iostream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <cstdlib>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

const float  FrameWidth = 720;							//�J��������̉摜��X  IEEE�J�����Ȃ�720*480
const float  FrameHeight = 480;							//�J��������̉摜��Y  USB�J�����Ȃ� 1280*720
const float  DispFrameWidth = 800;						//disp�̉�
const float  DispFrameHeight = 600;						//disp�̏c
int findLightSpan = 30;									//LED�T�����[�`�����s���t���[���Ԋu
int GreyThreshold = 83;								//2�l���̃X���b�V�����h
int  LightSpaceThreshold = 100;							//�������m�C�Y����臒l
int LightMoveThreshold = 70;							//�t���[�����ƂɈړ����������̋����������艺�Ȃ�Γ���̌����ƌ���
const int  LightMax = 7;								//�ő嗘�p�\�l��
const int  BinDataLong = 5;								//�o�C�i���f�[�^�̃r�b�g��
int  TtdLifetime = 30;									//Time to death�@LED��������Ȃ������Ƃ��ɑ�����ttd������ȏ�̂Ƃ��ALED�͏��������ƍl����
const int MaxAllowedIdMismatch = 5;						//ID�ɂ��G���[���m�̍ő�񐔁@����ȏ�Ȃ��Point��kill����
int ContourThickness = 1;								//�֊s���̑���
int  LineThickness = 10;								//����
int RedThreshold = 99;									//�ԐF��臒l
int BlueThreshold = 120;								//�F��臒l
int greenThreshold = 120;
int SpanHoughTransform = 30;							//�j���ϊ����s���t���[���Ԋu
const float  DifDisplayX = DispFrameWidth / FrameWidth;		// dispFrame/Frame �J�����摜����E�C���h�E�ւ̍��W�ϊ� xy���W�ɂ�����������disp�ł̍��W�ɂȂ�
const float  DifDisplayY = DispFrameHeight / FrameHeight;
const string WindowNameDisp = "Disp";

vector<Scalar> penColor = { Scalar(0, 0, 0), Scalar(150, 0, 0), Scalar(0, 150, 0), Scalar(0, 0, 150), Scalar(150, 150, 0) };  //�p�l���̕ω�����F�@ID=0�͍�
vector<Scalar> idColor = { Scalar(0, 150, 150), Scalar(150, 0, 0), Scalar(0, 150, 0), Scalar(0, 0, 150), Scalar(150, 150, 0) };  //ID�̐F

Mat rawCamera;												//�J����
Mat Thresholded2;											//rawcamera�̕\���p�R�s�[
Mat disp(Size(static_cast<int>(DispFrameWidth), static_cast<int>(DispFrameHeight)), CV_8UC3, Scalar(0, 0, 0));			//���C����ʁ@
Mat disp2(Size(DispFrameWidth, DispFrameHeight), CV_8UC3, Scalar(0, 0, 0));			//�J�[�\���ƍ�������Ď��ۂɕ\��������
Mat edgeImage(Size(DispFrameWidth, DispFrameHeight), CV_8UC1, Scalar(0));		//�G�b�W���o���ꂽ�C���[�W
vector<Mat> colorSplitDisp;														//臒l���ߗp��BGR�e�F�̐����݂̂𔲂��o����MAT

//for Ball
random_device rndBallDic;					//�{�[���̕������Y�p
int maxBallSpeed = 64;						//�{�[���̍ō����x
int defaultBallR = 20;						//�{�[���̃f�t�H���g���a
Scalar defaultBallCol = Scalar(0, 255, 0);	//�{�[���̃f�t�H���g�F��

//for PlayerBar
const int lineNum = 10;							//�ЂƂ�PlayerBar���\���������
const int lineThickness = 5;
const int PlayerBarRenewInterbal = 3;	
const int maxLineLength = 30;

//for pong
const int maxBall = 1;
const Point pointDisplayPos2Player(250, 80);		//��l�v���C���̃|�C���g�\���̏ꏊ
const Point pointDisplayPos3Player[3] = { Point(720, 560), Point(720, 100), Point(10, 330)}; //�O�l�v���C���̃|�C���g�\���ʒu
const int showPointScale = 3;
const int pointDispTime = 90;		//30hz 3s
const int maxContinuousBallreflect = 5;				//�{�[�������̉񐔈ȏ�A�����ăo�[�ƏՓ˂��Ă���Ȃ�{�[�����o�[�ň͂܂�ē����Ȃ��Ȃ��Ă���ƌ��Ĉꎞ�I�ɔ��˂𖳌��Ƃ���
int isPlaying = 0;					//pong�̃Q�[���Ƃ��Ă̏�ԕψڂ�����
int pongCnt = 0;					//pong�p�J�E���^�ϐ�
int winConditionPoint = 10;			//�����ɕK�v�ȓ_��
Mat fieldImage[2];					//pong�t�B�[���h�̔w�i�摜

//winSock2 variables
int const UDPReceivePort = 12345;		//UDP�Ŏ�M����|�[�g�ԍ�

struct collisionList{				//checkCollideCircleField�p
	int id=0;							//�t�B�[���h�ƏՓ˂����{�[����id
	double angle=0;					//�Փ˂����Ƃ��̒��S���猩�����W�A��
};
struct ballBarCollisionList{
	int ballId=0, barId=0;			//�{�[���ƃo�[��id
	Point2f vecOrigin;				//���˃x�N�g���̌��_
	Point2f vec;						//���˃x�N�g���̃x�N�g��
};
class PointerData{							//��ʂɕ\�������LED���_���Ǘ�����N���X

	private:

		int x, y, bin, l, buf,ttd;			//x axis, y axis , binary data, data length, LED�n�샋�[�`����LED����������Abin���X�V���ꂽ�� , buf:�O���bin��ۑ�, ttd:�������Ă���Point���炳�쏜�����܂ł̎���
		int allowedIdMis;					//ID�G���[���n�ɂ��G���[�̋�������
		bool alive, work;					//alive:LED���f�[�^�]������ work:��ʓ���LED�����݂��邩�@0:���݂��Ȃ��@1:���݂��AID�����肳��Ă���@4-2:���݂��邪ID�͌��蒆�@
		int id;								//���N��������id -1�Ŗ��m��
		int color;							//�J���[ID�@00:�����S���@01:�� 10:�@11:��
		bool cur;							//�J�[�\����� true:������Ă���
		int lx, ly;							//���O��xy���W
		vector<int> decidedat;				//id�����肷�邽�߂�3��ID��ǂݍ��݁A���������Ƃ�@
		int barIndex;						//�����O�o�b�t�@�̃C���f�b�N�X
	public:
		PointerData(){
			this->x = 0;
			this->y = 0;
			this->bin = 0;
			this->l = 0;
			this->buf = 0;
			this->ttd = 0;
			this->allowedIdMis = 0;
			this->alive = false;
			this->work = false;
			this->id = -1;
			this->color = -1;
			this->cur = false;
			this->lx = 0;
			this->ly = 0;
			this->barIndex = 0;
		}
		void newPoint(int x, int y){
			this->x = x;
			this->y = y;
			this->lx = 0;
			this->ly = 0;
			this->alive = true;
		}
		void killPoint(){
			this->x = 0;
			this->y = 0;
			this->bin = 0;
			this->l = 0;
			this->buf = 0;
			this->ttd = 0;
			this->allowedIdMis = 0;			
			this->alive = false;
			this->work = false;
			this->id = -1;
			this->color = -1;
			this -> decidedat.clear();
		
		}
		int getX(){
			return x;
		}
		int getY(){;
			return y;
		}
		int getBin(){
			return buf;
		}
		int getLength(){
			return l;
		}
		int addToBin(int dat){					//bin�Ƀf�[�^1,0������ �Ԃ�l: 0:����@1:ttd����̂��߃|�C���g�����@2:MaxAllowedIdMismatch�̏���̂��߃|�C���g����

			if (rawCamera.at<Vec3b>(this->y, this->x)[0] > BlueThreshold){		//�J�[�\���i�F�j�̌��m
				this->cur = true;
			}
			else{
				this->cur = false;
			}

			if (dat == 0){
				ttd++;	//�����Ȃ��ttd���C���N�������g
			}
			else{
				ttd = 0;
			}

			if (work == false){
				if (dat == 1){
					work = true;		//alive��false�����͂�1�Ȃ�Ύ�t��ԂƂ���
				}
			}
			else{						//alive=true�܂��t��
				bin = (bin << 1) + dat;
				l++;
				if (l > BinDataLong - 1){			//�f�[�^��l���S�f�[�^�ł���BinDataLong�܂肷�ׂẴf�[�^����M���I�����Ƃ��̏���
					work = false;
					buf = bin;
					bin = 0;
					l = 0;
					setIdColor();
				}
			}

			if (ttd > TtdLifetime){		//ttd������ȏ�Ȃ�΁ALED���E��
				killPoint();
				return 1;
			}
			/*
			if (allowedIdMis > MaxAllowedIdMismatch){	//ID�ɂ��G���[���w��񐔘A���Ō��m���ꂽ��Point���E��
			killPoint();
			}
			*/
			return 0;		//
		}
		void setXY(int x, int y){	
			this->lx = this->x;
			this->ly = this->y;
			this->x = x;
			this->y = y;
			if (this->x < 0) this->x = 0;
			if (this->x > FrameWidth-1) this->x = FrameWidth-1;		//x=720�̃s�N�Z���͑��݂��Ȃ�
			if (this->y < 0) this->y = 0;
			if (this->y > FrameHeight-1) this->y = FrameHeight-1;
		}
		bool getAlive(){
			return alive;
		}
		bool getWork(){
			return work;
		}
		int getTTD(){
			return ttd;
		}
		void clearTtd(){
			ttd = 0;
		}
		void setIdColor(){
			this->id = (this->buf >> 2);			//���2bit��id�Ƃ���
			this->color = (this->buf) & 3;			//����2bit��F�ԍ��Ƃ���
		}
		void incTtd(){ ttd++; }
		int checkTtdAndKill(){						//TTD���`�F�b�N���������𒴂�����Point���E��
			if (ttd > TtdLifetime){					//TTD�̍ő�l�ɒB������
				killPoint();						//�|�C���g���E��
				return -1;
			}
			return 0;
		}
		int getId(){
			return this->id;
		}
		int getColor(){
			return this->color;
		}
		bool getCur(){
			return cur;
		}
		void drawLine(){
			switch (this->color){
			case 1:
				line(disp, Point(static_cast<int>(DifDisplayX*this->lx), static_cast<int>(DifDisplayY*this->ly)), Point(static_cast<int>(DifDisplayX*this->x), static_cast<int>(DifDisplayY*this->y)), Scalar(0, 0, 255), LineThickness, 4, 0);
				break;
			case 2:
				line(disp, Point(static_cast<int>(DifDisplayX*this->lx), static_cast<int>(DifDisplayY*this->ly)), Point(static_cast<int>(DifDisplayX*this->x), static_cast<int>(DifDisplayY*this->y)), Scalar(255, 0, 0), LineThickness, 4, 0);
				break;
			case 3:
				line(disp, Point(static_cast<int>(DifDisplayX*this->lx), static_cast<int>(DifDisplayY*this->ly)), Point(static_cast<int>(DifDisplayX*this->x), static_cast<int>(DifDisplayY*this->y)), Scalar(0, 255, 0), LineThickness, 4, 0);
				break;
			case 4:
				line(disp, Point(static_cast<int>(DifDisplayX*this->lx), static_cast<int>(DifDisplayY*this->ly)), Point(static_cast<int>(DifDisplayX*this->x), static_cast<int>(DifDisplayY*this->y)), Scalar(0, 0, 0), LineThickness, 4, 0);
				break;
			default:
				break;
			}
		}
		void drawContourLine(){		//hough�ϊ��p�̗֊s�C���[�W��\������
			if (!cur) return;
			line(edgeImage, Point(static_cast<int>(DifDisplayX*this->lx), static_cast<int>(DifDisplayY*this->ly)), Point(static_cast<int>(DifDisplayX*this->x), static_cast<int>(DifDisplayY*this->y)), Scalar(255), ContourThickness, 4, 0);	
		}

		
		void drawCursor(){
			//cout << "this->x=" << to_string(DifDisplayX*this->x) << ",this->y=" << to_string(DifDisplayY*this->y) << endl;
			circle(disp2, Point(DifDisplayX*this->x, DifDisplayY*this->y), LineThickness + 1, Scalar(255, 255, 255), 2, 4, 0);
			/*
			if (id != -1){
				try{
					putText(disp2, to_string(id), Point(DifDisplayX*this->x, DifDisplayY*this->y), FONT_HERSHEY_COMPLEX, 0.5, Scalar(200, 200, 200));
				}
				catch (cv::Exception exp){ cout << "cv::exception" << endl; }
			}
			*/
			
				//cout << "col=" << to_string(color) << endl;
				switch (this->color){
				default:
					break;
				case 1:
					circle(disp2, Point(static_cast<int>(DifDisplayX*this->x), static_cast<int>(DifDisplayY*this->y)), LineThickness, Scalar(0, 0, 255), -1, 4, 0);
					break;
				case 2:
					circle(disp2, Point(static_cast<int>(DifDisplayX*this->x), static_cast<int>(DifDisplayY*this->y)), LineThickness, Scalar(255, 0, 0), -1, 4, 0);
					break;
				case 3:
					circle(disp2, Point(static_cast<int>(DifDisplayX*this->x), static_cast<int>(DifDisplayY*this->y)), LineThickness, Scalar(0, 255, 0), -1, 4, 0);
					break;
				case 4:
					circle(disp2, Point(static_cast<int>(DifDisplayX*this->x), static_cast<int>(DifDisplayY*this->y)), LineThickness, Scalar(0, 0, 0), -1, 4, 0);
					break;

				}
			
		}
		void setCur(bool in){
			this->cur = in;
		}
		void setId(int in){
			this->id = in;
		}
		void setColor(int inp){
			this->color = inp;
		}
};

class Ball{
private:
	float x,y,r;	//x���W,y���W,���a
	float lx, ly;	//
	float ax, ay;	//x,y���ւ̈ړ���
	Scalar col;		//�F
	int stat,id;	//1-�Ȃ�L�� id:���O�ɏՓ˂����o�[��id
	int refNum;		//�A�����Ĕ��ˏ��������t���[����
	
public:
	bool refR, refL, refU, refD;	//���˂����Ȃ�1

	Ball(){
		x = static_cast<int>(DispFrameWidth / 2);
		y = static_cast<int>(DispFrameHeight / 2);
		r = defaultBallR;
		ax = 4;
		ay = -3;
		col = defaultBallCol;
		stat = 0;
		refNum = 0;
		refR = false;
		refL = false;
		refU = false;
		refD = false;
		id = 0;
	}
	Ball(cv::Scalar ballcol) 
		:x(static_cast<int>(DispFrameWidth / 2)), y(static_cast<int>(DispFrameHeight / 2)), r(defaultBallR), ax(-3), ay(-5),
		col(ballcol), stat(0), refR(false), refL(false), refU(false), refD(false), id(0) {}

	void activate(){
		stat = 1;
	}
	void deactivate(){
		stat = 0;
	}
	void setDafault(){
		x = static_cast<int>(DispFrameWidth / 2);
		y = static_cast<int>(DispFrameHeight / 2);
	}
	void move(){		//ax,ay�����{�[���𓮂���
		//cout << "ax:" << ax << "ay:" << ay << endl;
		refR = false;
		refL = false;
		refU = false;
		refD = false;
		int tx=lx, ty=ly;		//���O��xy
		lx = x, ly = y;
		x = x + ax;
		y = y + ay;
		if ((x - r < 0)){		//x���̉�ʒ[�Փ�
			x = lx;
			lx = tx;
			ax = -ax;
			refL = true;
		} 
		if (x + r > DispFrameWidth){	//�E�Փ�
			x = lx;
			lx = tx;
			ax = -ax;
			refR = true;
		}
		if ((y - r < 0)){		//y���̉�ʒZ�Փ�
			y = ly;
			ly = ty;
			ay = -ay;
			refU = true;
		}
		if (y + r > DispFrameHeight){
			y = ly;
			ly = ty;
			ay = -ay;
			refD = true;
		}
	}
	void setAccel(float pax,float pay){
		ax = pax;
		ay = pay;
	}
	void setAccelX(float pax){
		ax = pax;
	}
	void setAccelY(float pay){
		ay = pay;
	}
	void setPos(int px, int py){
		x = px;
		y = py;
	}
	void setColor(Scalar inpCol){
		col = inpCol;
	}
	void incRefNum(){ refNum++; }
	void clearRefNum() { refNum = 0; }
	float getX(){
		return x;
	}
	float getY(){
		return y;
	}
	float getR(){
		return r;
	}
	float getAX(){
		return ax;
	}
	float getAY(){
		return ay;
	}
	int getId(){ return id; }
	int getRefNum(){ return refNum; }
	float getLx(){ return lx; }
	float getLy() { return ly; }
	void setId(int inp) { id = inp; }
	void draw(Mat dest){
		if (stat == 1) {
			circle(dest, Point(x, y), r, col,-1);
		}
	}
};
class PlayerBar{
private:
	int ringBufIndex;	//Bar���i�[����z��̃����O�o�b�t�@�I�擪�̗v�f
	float barArray[lineNum+1][2];
	Scalar color;		//�o�[�̐F�@idColor�Q��
	int stat,point;			//1�ɓ���
						//id���K�v,,��M����id�ɏ]���Đ��̐F��ς��邽�߂�
public:
	PlayerBar(){
		color = Scalar(0,255,0);
		for (int i = 0; i < lineNum + 1; i++){
			barArray[i][0] = 0;
			barArray[i][1] = 0;
		}
		point = 0;
	}
	PlayerBar(Scalar plyCol){
		ringBufIndex = 0;
		color = plyCol;
		stat = 0;
		point = 0;
		for (int i = 0; i < lineNum + 1; i++){
			barArray[i][0] = 0;
			barArray[i][1] = 0;
		}
	}
	void activate(){
		stat = 1;
	}
	void deactivate(){
		stat = 0;
	}
	void addBar(int px, int py){		//���_��ǉ�
		barArray[ringBufIndex][0] = DifDisplayX*px;
		barArray[ringBufIndex][1] = DifDisplayY*py;
		ringBufIndex = (ringBufIndex + 1) % (lineNum + 1);		//�����O�o�b�t�@��i�߂�
	}
	void addSlowBar(int px, int py){		//���_��ǉ�

		int prvInd = (ringBufIndex - 1) % (lineNum + 1);
		int apx = barArray[prvInd][0], apy = barArray[prvInd][1];		//�ЂƂO�̒��_���W
		int s2;															//�����̓��
		if (apx *apy != 0){		//�O�̍��W�������l�ł͂Ȃ���
			s2 = ((px - apx) ^ 2 + (py - apy) ^ 2);
			if (s2 < maxLineLength^2){
				barArray[ringBufIndex][0] = px;
				barArray[ringBufIndex][1] = py;
			}
			else{		//maxLineLength�ȏ�̐���`���Ȃ�
				barArray[ringBufIndex][0] = maxLineLength*(px-apx) / sqrt(s2) + apx;
				barArray[ringBufIndex][1] = maxLineLength*(py-apy) / sqrt(s2) + apy;			//�Â��_����V�����_�ւ̃x�N�g���𐳋K�����āAmaxLIneLength�̒����̃x�N�g���𐶐�
			}
		}
		ringBufIndex = (ringBufIndex + 1) % (lineNum + 1);		//�����O�o�b�t�@��i�߂�
	}
	void draw(Mat dest){
		if (stat = 1){
			for (int i = 0; i < lineNum + 0; i++){
				float ax = barArray[(i + ringBufIndex) % (lineNum + 1)][0], ay = barArray[(i + ringBufIndex) % (lineNum + 1)][1], bx = barArray[(i + ringBufIndex + 1) % (lineNum + 1)][0], by = barArray[(i + ringBufIndex + 1) % (lineNum + 1)][1];
				if (ax*ay*bx*by!=0) line(dest, Point(ax, ay), Point(bx,by ), color,LineThickness); //���W��0�̒��_�������͕`���Ȃ�
			}
		}
	}
	bool getCollideBallPos(Ball& obj,Point2f& poi,Point2f& newpos){		//ball�ƏՓ˂��Ă��邩,�Փ˂��Ă����甽�˃x�N�g����Ԃ� obj:�{�[���������@poi: newpos:ball��bar�ɂ߂肱�݂𒼂������ball�̍��W http://marupeke296.com/COL_2D_No5_PolygonToCircle.html
		float ballx=obj.getX(),bally=obj.getY(), ballr=obj.getR();
		float ary1x, ary1y, ary2x, ary2y;
		float sx, sy, ax, ay,bx,by,absSA,absS,d,dotas,dotbs,sbr;			//d=|S�~A|/|S| S�͏I�_-�n�_
		float nx, ny,nabs;													//�Փː��̐��K���@���x�N�g��
		float fx=obj.getAX(), fy=obj.getAY();								//�Փˎ��̃x�N�g��
		float a;
		float rx, ry;
		bool ret=false;														//�Ԃ�l
		for (int i = 0; i < lineNum + 0; i++){
			ary2x = barArray[(i + ringBufIndex + 1) % (lineNum + 1)][0];
			ary2y = barArray[(i + ringBufIndex + 1) % (lineNum + 1)][1];
			ary1x = barArray[(i + ringBufIndex) % (lineNum + 1)][0];
			ary1y = barArray[(i + ringBufIndex) % (lineNum + 1)][1];
			if ((ary1x*ary2x == 0) || (ary2x*ary2y == 0)) continue;  //���W��(0,0)���܂ޏꍇ�v�Z�ɓ���Ȃ�
			sx = ary2x - ary1x;								//S =ary2-ary1
			sy = ary2y - ary1y;
			ax = ballx - ary1x;
			ay = bally - ary1y;
			absSA = abs(sx*ay-ax*sy);
			absS = sqrt(sx*sx + sy*sy);
			if (absS == 0) continue;
			d = absSA / absS;
			//cout << d << endl;
			if (d > ballr){
				continue;
			}
			else{
				bx = ballx - barArray[(i + ringBufIndex + 1 ) % (lineNum + 1)][0];
				by = bally - barArray[(i + ringBufIndex + 1 ) % (lineNum + 1)][1];
				if ((ax*sx + ay*sy)*(bx*sx + by*sy) > 0){
					continue;
				}
				else{			//�Փˁ@���˃x�N�g�������߂�    n=S�~A=(sx,sy,0)�~(sx,sy,1)=(sy,-sx,0)
					nabs = sqrt(sy*sy + sx*sx);			//n�̐�Βl
					nx = sy / nabs;
					ny = -sx / nabs;
					//r=f-2dot(f,n)*n
					//a=2(fxnx+fyny)
					a = 2*(fx*nx + fy*ny);
					rx = fx - a*nx;
					ry = fy - a*ny;
					if (rx*rx + ry*ry > maxBallSpeed*maxBallSpeed){			//�x�N�g���̍ő呬�x�����@maxBallSpeed�܂�
						float sum_r = sqrt(rx*rx + ry*ry);
						rx = rx / sum_r*maxBallSpeed;
						ry = ry / sum_r*maxBallSpeed;
					}
					//�߂荞�񂾉~�������͂����@�@����S�̖@���� (Sx,Sy,0)�~(0,0,1)=(-sy,sx,0)
					//���̖@���𐳋K�������nx=1/��(sx^2+sy^2)*-sy, ny=1/��(sx^2+sy^2)*sx
					//S�ɑ΂���,S�̍�������(bx,by)�ւ̃x�N�g�����E�ɂ��邪���ɂ��邩��,(Sx,sy)�~(BX,BY)>0�Ȃ�n*+ <0 �Ȃ�n*-
					//
					sbr = sx*bally - sy*ballx;
					newpos = Point2f(obj.getLx(), obj.getLy());			//�Փ˂��钼�O�̍��W��Ԃ�
					/*				���˃x�N�g����-1*�𑫂��Ă߂荞�񂾃{�[���������߂������@���܂������Ȃ�
					if (sbr > 0){
						newpos = Point2f((ballr - d)*nx+ballx, (ballr - d)*ny+bally);
						cout << "sbr=+" << endl;
					}
					else{
						newpos = Point2f((ballr - d)*-nx+ballx, (ballr - d)*-ny+bally);
						cout << "sbr=-" << endl;
					}
					*/
					poi=Point2f(rx, ry);
					
					cout << "d=" << d << ":from(" << fx << "," << fy << ") to (" << fx - a*nx << "," << fy - a*ny << ")" << endl;
					cout << "pos:(" << ballx << "," << bally << ") to (" << ballx + newpos.x << "," << bally + newpos.y << ")" << endl;
					cout << "----------------------------------------------------------------------------------------------------------" << endl;
					return true;
				}
			}
		}
		return false;
	}
	void addOneRingBuf(){			//�����O�o�b�t�@�̃C���f�b�N�X��1�i�߂�
		ringBufIndex = (ringBufIndex + 1) % (lineNum + 1);		//�����O�o�b�t�@��i�߂�
	}
	void addPoint(int ip){
		point = point + ip;
		if (point < 0) point = 0;
	}
	int getPoint(){
		return point;
	}
	void setPoint(int ip){
		point = ip;
	}	
	void setColor(Scalar colInp){ color = colInp; }
	void reset(){
		for (int i = 0; i < lineNum + 1; i++){
			barArray[i][0] = 0;
			barArray[i][1] = 0;
		}
	}
};
class Pong{
private:
	vector<Ball> b;			//�{�[���̐�
	vector<PlayerBar> p;		//�v���C���[��			�������ԈႦ�Ă���ABall.draw����Pong.ballNum��ǂނ��Ƃ��ł��Ȃ��ABall�̐���Pong�ł͂Ȃ�Ball�����ׂ�
	int stat, playerNum, ballNum;	//playerNum:�v���C�l�� ballNum:�{�[����
public:
	Pong(int ply, int mball) :playerNum(ply), ballNum(mball), stat(0){
		for (int i = 0; i < playerNum; i++) p.push_back(PlayerBar(idColor[i]));	//vector<Ball>�̐錾
		for (int i = 0; i < ballNum; i++) b.push_back(Ball());
	}
	~Pong(){}
	void startGame(){	//�Q�[�����n�߂�
		//�{�[���𐶎Y
		for (int i = 0; i < ballNum; i++){
			b[i].activate();
		}
		//�Q�[���o�[�𐶎Y
		for (int i = 0; i < playerNum; i++){
			p[i].activate();
		}
		stat = 1;		//�Q�[����
	}
	void endGame(){
		//�{�[����񊈐���
		for (int i = 0; i < ballNum; i++){
			b[i].deactivate();
		}
		//�Q�[���o�[��񊈐���
		for (int i = 0; i < playerNum; i++){
			p[i].deactivate();
		}
		stat = 0;
	}
	void moveBalls(){								//�{�[�������ׂē�����
		if (stat == 1){
			for (int i = 0; i < ballNum; i++){
				b[i].move();
			}
		}
	}
	void updateBars(vector<PointerData>& source, bool isSlow){		//Point Source[LightMax]�����ɂ��ׂẴv���C���[�o�[���X�V
		for (int i = 0; i < playerNum; i++){
			float srcx = source[i].getX(), srcy = source[i].getY();		//�o�[�̐V�������W���擾
			if ((srcx != 0) || (srcy != 0)){							//���W��(0,0)�łȂ���
				p[i].setColor(idColor[source[i].getId()]);
				if (isSlow){
					p[i].addSlowBar(srcx, srcy);
				}
				else{
					p[i].addBar(srcx, srcy);
				}
			}
			else{
				p[i].addOneRingBuf();
			}
		}
	}

	vector<ballBarCollisionList> checkPlayerBallCollide(){					//���ׂẴo�[�Ń{�[���Ƃ̏Փ˂𔻒肵�A�Փ˂��Ă����甽�ˏ��� vector<vector<int>>{�Փ˂���Bar��id,�Փ˂���Ball��id}��Ԃ�
		Point2f poi, newpos, correctedPos;
		vector<ballBarCollisionList> listCollide;
		ballBarCollisionList temp;
		for (int j = 0; j < ballNum; j++){
			bool isRefrectinFrame = false;									//���̃t���[���ł��̃{�[���͔��ˏ������s�������H
			for (int i = 0; i < playerNum; i++){			
				if (p[i].getCollideBallPos(b[j], poi, newpos)){
					cout << "collide:ball[" << to_string(j) << "]" << endl;
					isRefrectinFrame = true;									//���̃{�[���͂��̃t���[���Ŕ��˂��s����
					if (b[j].getRefNum()<maxContinuousBallreflect){				//maxContinuousBallrefrect�̉񐔂����A�����Ĕ��ˏ��������Ă��Ȃ����
						b[j].incRefNum();										//refNum���C���N�������g
						b[j].setAccel(poi.x, poi.y);
						b[j].setPos(newpos.x, newpos.y);
						temp.ballId = j;
						temp.barId = i;
						temp.vecOrigin = correctedPos;
						temp.vec = Point(newpos.x, newpos.y);
						listCollide.push_back(temp);
						b[j].setId(i);							//�{�[���ɒ��O�ɏՓ˂����v���C���[id�����
					}
					else{
						cout << "ball " << to_string(j) << " skiped refrect precedure due to max refNum" << endl;
					}
				}
				else{

				}
				
			}
			if (isRefrectinFrame == true){ b[j].incRefNum(); }
			else { b[j].clearRefNum(); }
		}
		return listCollide;
	}
	void changeAllBallColor(vector<ballBarCollisionList>& listCollide){ //correctCollide()����A���Ă���vector<vector<int>>�����ƂɁA�{�[���̐F�𒵂˕Ԃ����v���C��̐F�ɕύX����

		for (auto table : listCollide){
			cout << "hit by ballid=" << to_string(table.ballId) << endl;;
			b[table.ballId].setId(table.barId);
			b[table.ballId].setColor(idColor[table.barId]);
		}
	}
		vector<collisionList> checkCollideWithCircleField(){				//�t�B�[���h���S����r=300�̃t�B�[���h��ݒ肵�A����ƏՓ˂����{�[����id�ƒ��S����̊p�x��Ԃ��B
			vector<collisionList> collideList;							//�Ԃ�l��{�Փ˂����{�[����id,X��+�����玞�v���ɂƂ����p�x��}						
			float fieldCtrX = DispFrameWidth / 2.0;
			float fieldCtrY = DispFrameHeight / 2.0;
			int ind = 0;												//�z��ϐ�b�̓Y����,�͈�for�ł��Y�������擾�ł��Ȃ��̂ŗ͋Z��
			for (auto bb : b){
				int r2_a = static_cast<int>(bb.getX() - fieldCtrX);
				int r2_b = static_cast<int>(bb.getY() - fieldCtrY);
				int r2_ball = pow(r2_a, 2) + pow(r2_b, 2);
				if (r2_ball > pow((DispFrameHeight / 2) - defaultBallR * 2, 2)){
					collisionList temp;
					temp.id = ind;
					temp.angle = atan2(r2_b, r2_a);
					collideList.push_back(temp);
					//cout << "r2_a=" << to_string(r2_a) << ":r2_b=" << to_string(r2_b) << ":r=" << to_string(r2_ball) + ":deg=" + to_string((atan2(r2_b , r2_a))/M_PI*180) << endl;
				}
				ind++;
			}
			return collideList;
		}


		void draw(Mat dest){
			//�{�[����`��
			for (auto bb : b){
				bb.draw(dest);
			}
			//�Q�[���o�[��c
			for (auto pp : p){
				pp.draw(dest);
			}
		}
		void displayPlayerPoint(Mat& dest){		//���_��\�� 
			switch (playerNum){
			case 2:
				putText(dest, to_string(p[1].getPoint()) + " - " + to_string(p[0].getPoint()), pointDisplayPos2Player, FONT_HERSHEY_COMPLEX, showPointScale, Scalar(255, 255, 255), 10);
				break;
			case 3:
				putText(dest, to_string(p[0].getPoint()), pointDisplayPos3Player[0], FONT_HERSHEY_COMPLEX, 3, Scalar(255, 255, 255), 2);
				putText(dest, to_string(p[1].getPoint()), pointDisplayPos3Player[1], FONT_HERSHEY_COMPLEX, 3, Scalar(255, 255, 255), 2);
				putText(dest, to_string(p[2].getPoint()), pointDisplayPos3Player[2], FONT_HERSHEY_COMPLEX, 3, Scalar(255, 255, 255), 2);
				break;
			default:
				break;
			}
		}
		void dispWinner(int win, Mat& dest){ //win�Ԗڂ̃v���C���[��winner�Ƃ��ĕ\��
			cv::putText(disp2, "Player " + to_string(win) + " WIN", Point(300, 300), FONT_HERSHEY_COMPLEX, 2, Scalar(200, 200, 200), 5);
		}
		void addBallScore(int id, int scr){
			p[id].addPoint(scr);
		}
		void setBallScore(int id, int scr){
			p[id].setPoint(scr);
		}
		void clearAllPoint(){
			for (auto pp : p){
				pp.setPoint(0);
				cout << "point=" << to_string(pp.getPoint()) << endl;
			}
		}
		bool checkBallHitLeftWall(int ballid){
			return b[ballid].refL;
		}
		bool checkBallHitRightWall(int ballid){
			return b[ballid].refR;
		}
		void resetBallPos(int ballid){
			b[ballid].setDafault();
		}
		int getPlayerbarPoint(int idp){
			return p[idp].getPoint();
		}
		void activeBall(int ballid){
			b[ballid].activate();
		}
		void deactiveBall(int ballid){
			b[ballid].deactivate();
		}
		void setBallInitVec(int ballid, Point2f vec){
			b[ballid].setAccelX(vec.x);
			b[ballid].setAccelY(vec.y);
		}
		int getPlayerNum(){
			return playerNum;
		}
		int getBallId(int id){ return b[id].getId(); }
		void setBallId(int idp, int s){ b[idp].setId(s); }
	};
	// ���l���Q�i��������ɕϊ�
	string to_binString(unsigned int val){
		if (!val)
			return std::string("0");
		std::string str;
		while (val != 0) {
			if ((val & 1) == 0)  // val �͋������H
				str.insert(str.begin(), '0');  //  �����̏ꍇ
			else
				str.insert(str.begin(), '1');  //  ��̏ꍇ
			val >>= 1;
		}
		return str;
	};

	int main(int argc, char *argv[]){

		int key = 0;		//key�͉����ꂽ�L�[
		int loopTime = 0;
		int frame = 0;										//�o�߃t���[��
		int winnerId = 0;									//�����҂�id
		bool isDebug = false;
		vector<ballBarCollisionList> PlayerBallCCollideListdebug;	//�{�[���ƃv���C���[�o�[�̏Փ˃��X�g

		//winSock2 object
		WSAData wsadat;
		SOCKET sock;
		struct sockaddr_in addr;
		u_long isnBlock = 1;					//WSAD���A���u���b�L���O�ŉ^�p���邽�߂̃t���O
		char buf[2048] = {};					//��M����UDP�f�[�^���i�[����o�b�t�@
		int recvLen;							//��M�������b�Z�[�W�̒���

		int dummy;
		if (WSAStartup(MAKEWORD(2, 0), &wsadat) != 0){
			std::cout << "winSock2 intialization failed!" << std::endl;
			return -1;
		}
		std::cout << "winSock2 intialized succesfully!" << std::endl;
		std::cout << "version=" << std::to_string(wsadat.wVersion) << std::endl;

		sock = socket(AF_INET, SOCK_DGRAM, 17);

		addr.sin_family = AF_INET;
		addr.sin_port = htons(UDPReceivePort);
		addr.sin_addr.S_un.S_addr = INADDR_ANY;

		bind(sock, (struct sockaddr *)&addr, sizeof(addr));
		ioctlsocket(sock, FIONBIO, &isnBlock);		//�m���u���b�L���O���[�h

		//opencv 
		vector<PointerData> PointData(LightMax);					//vector�I�u�W�F�N�g�g����!!
		VideoCapture cap(0);
		cap.set(CV_CAP_PROP_FRAME_WIDTH, FrameWidth);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, FrameHeight);

		cout << "Initializing\n";

		if (!cap.isOpened()) {
			std::cout << "failed to capture camera";
			return -1;
		}

		namedWindow(WindowNameDisp, CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
		namedWindow( "Thresholded2", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
		namedWindow("blueImage", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
		namedWindow("greenImage", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
		namedWindow("redImage", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);


		while (1){

			double f = 1000.0 / cv::getTickFrequency();		//�v���O��������J�n�����frame�����擾

			int64 time = cv::getTickCount();

			while (!cap.grab());
			cap.retrieve(rawCamera);

			cvtColor(rawCamera, Thresholded2, CV_BGR2GRAY);		//�O���C�X�P�[����
			threshold(Thresholded2, Thresholded2, GreyThreshold, 255, CV_THRESH_BINARY);

			//�t���[���̈ꕔ��{�����āA�V����LED����������B������10����
			if (frame%findLightSpan){
				for (int y = 0; y < FrameHeight; y += 50){
					const uchar *pLine = Thresholded2.ptr<uchar>(y);
					for (volatile int x = 0; x < FrameWidth; x += 50){
						if (pLine[x] > GreyThreshold){
							bool isNew = true;					//true if LED is found newly
							for (int pointNum = 0; pointNum < LightMax - 0; pointNum++){
								if ((abs(PointData[pointNum].getX() - x) < 100) & (abs(PointData[pointNum].getY() - y) < 100) & (PointData[pointNum].getAlive())){
									isNew = false;
									break;
								}
							}

							if (isNew){
								for (int PointNum = 0; PointNum < LightMax - 0; PointNum++){
									if (!PointData[PointNum].getAlive()){
										PointData[PointNum].newPoint(x, y);		//generator new LED
										cout << "new LED at(" << to_string(x) << "," << to_string(y) << ") -> Point[" << to_string(PointNum) << "]" << endl;
										break;
									}
								}
							}
						}
					}
				}
			}

			//���ׂĂ�Point���d�S��p���Ĉʒu���̍X�V������
			//���̌�A�J�[�\���ʒu��RGB��������ID�����肷��


			double gx, gy;
			Moments moment;

			for (int pointNum = 0; pointNum < LightMax - 0; pointNum++){		//check each LED on or off
				if (PointData[pointNum].getAlive()){							//�w�肵��point�������Ă���Ȃ�

					int cutposx = PointData[pointNum].getX() - LightMoveThreshold, cutposy = PointData[pointNum].getY() - LightMoveThreshold;	//cutpos�̎w����W����ʊO�ɂȂ��ăG���[�͂��̂�h��
					if (cutposx < 0) cutposx = 0;
					if (cutposx > FrameWidth - 2 * LightMoveThreshold - 1) cutposx = FrameWidth - 2 * LightMoveThreshold - 1;
					if (cutposy < 0) cutposy = 0;
					if (cutposy > FrameHeight - 2 * LightMoveThreshold - 1) cutposy = FrameHeight - 2 * LightMoveThreshold - 1;
					Mat cut_img(Thresholded2, cvRect(cutposx, cutposy, 2 * LightMoveThreshold, 2 * LightMoveThreshold));				//LED���_���ӂ�؂���

					moment = moments(cut_img, 1);																						//�؂�����cut_img�ŐV�������[�����g���v�Z
					gx = moment.m10 / moment.m00;																						//�d�S��X���W
					gy = moment.m01 / moment.m00;																						//�d�S��y���W

					if ((gx >= 0) && (gx <= 2 * LightMoveThreshold) && (gy >= 0) && (gy <= 2 * LightMoveThreshold)){					//gx,gy��cut_img�͈̔͂���͂ݏo���悤�Ȉُ�Ȓl���H

						int newX = PointData[pointNum].getX() + (int)(gx)-LightMoveThreshold;												//gx,gy�ɂ��X�V���ꂽXY���W��
						int newY = PointData[pointNum].getY() + (int)(gy)-LightMoveThreshold;
						bool isDub = false;																								//�X�V���ꂽ���W�͂ق���point�Ƃ��Ԃ��Ă��Ȃ����H
						for (int pointNumToCheckDublication = 0; pointNumToCheckDublication < pointNum; pointNumToCheckDublication++){
							if ((abs(PointData[pointNumToCheckDublication].getX() - newX) < LightMoveThreshold) && (abs(PointData[pointNumToCheckDublication].getY() - newY) < LightMoveThreshold)){
								isDub = true;		//��������LED�͂��łɂ���PointData�̂����̂ЂƂł���
								break;
							}
						}

						if (!isDub){									//�ق���PointData�Əd�����Ȃ����Ƃ�����Ċm�F�ł�����
							PointData[pointNum].setXY(newX, newY);		//XY���W���X�V	
						}
					}
					else{

					}

					//��,��LED����id��ǂݍ��ޏ���
					int id = 0;		
					if (rawCamera.at<Vec3b>(PointData[pointNum].getY() | 1, PointData[pointNum].getX())[2] > RedThreshold){		//�ԐFLED�͓_�����Ă��邩�H
						id |= 1 << 0;
					}
					if (rawCamera.at<Vec3b>(PointData[pointNum].getY() | 1, PointData[pointNum].getX())[1] > greenThreshold){		//�ΐFLED�͓_�����Ă��邩�H
						id |= 1 << 1;
					}
					if (rawCamera.at<Vec3b>(PointData[pointNum].getY() | 1, PointData[pointNum].getX())[0] > BlueThreshold){		//�FLED�͓_�����Ă��邩�H
						id |= 1 << 2;
					}
					id = id - 1;
					if (id == -1) {
						PointData[pointNum].incTtd();						//TTD���C���N�������g����
						//cout << "pointData.ttd=" << to_string(PointData[pointNum].getTTD()) << endl;
						if (PointData[pointNum].checkTtdAndKill()==1){				//TTD���ő吔�ɒB������PointData��kill()����
							cout << "Point[" << to_string(pointNum) << "] killed by TTD" << endl;
						}
					}
					else{
						PointData[pointNum].clearTtd();																	//��ʏ�Ɍ��N�����������݂��Ă���Ȃ�ttd���N���A
					}
					PointData[pointNum].setId(id);			//�ǂݎ����id���Z�b�g
				}
				else{

				}
				PointData[pointNum].drawLine();
			}
			/*
			//debug
			for (auto point : PointData){
				cout << to_string(point.getId()) << ":col=" << to_string(point.getColor) << endl;
			}
			*/

			//DNS�p�P�b�g����M���A�f�[�^��PointData�Ɋi�[����
			recvLen = recv(sock, buf, sizeof(buf), 0);
			while (recvLen > 0){
				int recvId, recvCol;
				cout << to_string(recvLen) << "UDP incomming" << endl;
				char recvdDat = buf[0];			//buf��������Ƃ��Â���M�f�[�^��ǂݎ��
				recvId = recvdDat >> 3;
				recvCol = recvdDat & 7;
				cout << "incoming(ID=" << to_string(recvId) << ",Col=" << to_string(recvCol) << ")" << endl;
				for (int pId = 0; pId < LightMax;pId++){
					if (PointData[pId].getId() == recvId){
						PointData[pId].setColor(recvCol);
						PointData[pId].setId(recvId);
							cout << "saved to Point[" << to_string(PointData[pId].getId()) << "]" << endl;
					}				
				}
				recvLen--;
			}

			disp2 = disp.clone();	//disp2<-disp�R�s�[

			for (int pointNum = 0; pointNum < LightMax - 0; pointNum++){		//display bin to 

				//putText(disp2, "(x,y)=(" + to_string(PointData[pointNum].getX()) + "," + to_string(PointData[pointNum].getY()) + ")" + "id=" + to_string(PointData[pointNum].getId()) + ":color=" + to_string(PointData[pointNum].getColor()) + "pointData[" + to_string(pointNum) + "]", Point(DifDisplayX*PointData[pointNum].getX(), DifDisplayY*PointData[pointNum].getY()), FONT_HERSHEY_COMPLEX, 0.5, Scalar(200, 200, 200));
				putText(disp2, "id=" + to_string(PointData[pointNum].getId()) + ":color=" + to_string(PointData[pointNum].getColor()), Point(DifDisplayX*PointData[pointNum].getX(), DifDisplayY*PointData[pointNum].getY()), FONT_HERSHEY_COMPLEX, 0.5, Scalar(200, 200, 200));
				PointData[pointNum].drawCursor();
			}

			if (loopTime > 33){		//�t���[�����[�g�\��
				putText(disp2, "fps=" + to_string(loopTime), Point(DispFrameWidth - 80, DispFrameHeight - 35), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 200));
			}
			else{
				putText(disp2, "fps=" + to_string(loopTime), Point(DispFrameWidth - 80, DispFrameHeight - 35), FONT_HERSHEY_COMPLEX, 0.5, Scalar(200, 200, 200));
			}

			putText(Thresholded2, "GreyThreshold=" + to_string(GreyThreshold) + ":RedThreshold=" + to_string(RedThreshold) + ":BlueThreshold=" + to_string(BlueThreshold) + ":greenThreshold=" + to_string(greenThreshold) + ":frame=" + to_string(frame) + ":isPlaying=" + to_string(isPlaying), Point(0, 10), FONT_HERSHEY_COMPLEX, 0.5, Scalar(200, 200, 200));

			if (isDebug){
				split(rawCamera, colorSplitDisp);
				putText(colorSplitDisp[0], ":BlueThreshold=" + to_string(BlueThreshold), Point(0, 10), FONT_HERSHEY_COMPLEX, 0.5, Scalar(200, 200, 200));
				putText(colorSplitDisp[1], ":GreenThreshold=" + to_string(greenThreshold), Point(0, 10), FONT_HERSHEY_COMPLEX, 0.5, Scalar(200, 200, 200));
				putText(colorSplitDisp[2], ":RedThreshold=" + to_string(RedThreshold), Point(0, 10), FONT_HERSHEY_COMPLEX, 0.5, Scalar(200, 200, 200));
				threshold(colorSplitDisp[0], colorSplitDisp[0], BlueThreshold, 255, CV_THRESH_BINARY);
				threshold(colorSplitDisp[1], colorSplitDisp[1], greenThreshold, 255, CV_THRESH_BINARY);
				threshold(colorSplitDisp[2], colorSplitDisp[2], RedThreshold, 255, CV_THRESH_BINARY);
				cv::imshow("blueImage", colorSplitDisp[0]);
				cv::imshow("greenImage", colorSplitDisp[1]);
				cv::imshow("redImage", colorSplitDisp[2]);
			}
			cv::imshow(WindowNameDisp, disp2);
			cv::imshow("Thresholded2", Thresholded2);
			//cv::imshow("edgeImage", edgeImage);

			key = waitKey(1);

			if (key == 'q'){					//�I��
				destroyWindow(WindowNameDisp);
				return 0;
			}

			if (key == 'c'){
				disp = Scalar(0, 0, 0);			//disp��������
			}
			if (key == 'w'){					//greyThreshold +1
				if (GreyThreshold + 1 < 256) GreyThreshold++;
			}

			if (key == 's'){					//greyThreshold -1
				if (GreyThreshold - 1 >= 0) GreyThreshold--;
			}

			if (key == 'e'){					//RedThreshold +1
				if (RedThreshold + 1 < 256) RedThreshold++;
			}

			if (key == 'd'){					//RedThreshold -1wwwi
				if (RedThreshold - 1 >= 0) RedThreshold--;
			}
			if (key == 'r'){					//BlueThreshold +1
				if (BlueThreshold + 1 < 256) BlueThreshold++;
			}

			if (key == 'f'){					//blueThreshold -1
				if (BlueThreshold - 1 >= 0) BlueThreshold--;
			}
			if (key == 't'){					//blueThreshold -1
				if (greenThreshold + 1 < 256) greenThreshold++;
			}
			if (key == 'g'){					//blueThreshold -1
				if (greenThreshold - 1 >= 0) greenThreshold--;
			}
			if (key == 'h') {					//�f�o�b�O���[�h
				isDebug = !isDebug;
			}
			loopTime = (cv::getTickCount() - time)*f;
			frame++;
		}
	}

