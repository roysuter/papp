/**************************************************************************************
(c)2014, Pierre RAUFAST. Visit http://thinkrpi.wordpress.com
Software Magician
This file controls robot's eyes. 
It uses RaspiCamCV library made by Emil Valkov, based on a Pierre Raufast project.

version 2014 03 30 with eye blink
***************************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>
#include <math.h> // not used only for sqrt
#include <time.h>
#include <unistd.h>

#include <cv.h>
#include <highgui.h>

// OPENCV include
#include <iostream>
#include <fstream>
#include <sstream>
#include "time.h"

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
// #include "/home/pi/pierre/libfacerec-0.04/include/facerec.hpp"

using namespace cv;
using namespace std;
// end OPENCV

// library with access to webcam
#include "RaspiCamCV.h"

typedef struct
{
	Point pos;	// position of face
	int off;	// number of image where visitors is gone
	int size; 	// size of the face
	Point servo;	// position of servo horizontal and vertical 
} VISITORS;


#define MAX_VISITOR	10
#define SERVO_H	0
#define SERVO_V	1
#define SERVO_B	2


// global variables
int nTrace;				// are you in verbose mode or not ? (>1=verbose)
int g_nMaxFrameOff; 	// number of images without visitor meaning he left
int g_nToleranceServo; 	// distance max without move from eye
int g_nMax_Frames_Looked; // number frames max before changing the guy looked
int g_nExVisitors;		// number of Ex visitors
int g_nVisitors;		// number of current visitors
int b_newVisitors;		// new visitor detected ?
int g_servoMax[3];			// amplitude max of servo
int g_servoMin[3];			// amplitude min of servo
int b_reverseServo[3];		// reverse servo 
int g_blinkPeriod;			// time between two blink eye (in seconds)
int g_halfBlinkDuration;	// time to close the eye
int g_napPeriod;			// period btw 2 short nap=eyes closed (in seconds)
int g_napDuration;			// duration of a nap (in seconds)
int b_napOngoing;			// i am during a nap ? 
int b_servoTrace;			// trace for all servo movement ? 
int g_nFreqBehavior;		// frequence of special behavior (0-1000)

VISITORS lastLookedVisitor;				// last new visitor 
VISITORS noVisitor; 					// no visitor : look center
VISITORS exVisitors[MAX_VISITOR+1];		// position of previous spectators
VISITORS currentVisitors[MAX_VISITOR]; 	// position of existing visitors
RaspiCamCvCapture * capture;
CascadeClassifier face_cascade; // pattern to detect the face 
//CascadeClassifier eye_cascade; 	// pattern to detect eyes
Point previousServo;			// previous servo position
Mat imageMat;
int oldServoPosition[2];		// old servo position : to 

// for eye blink
double lastBlinkTime;
double lastNapTime;

FILE * servoDev;		//point to /dev/servoblaster (servoblaster daemon required)

//
// in verbose mode, display a trace txt
//
void trace(char* sText)
{
	if (nTrace==1){printf("%s",sText);}
}


/////////////////////////////////////////////////////////////////////////
// Set Servo movement
// ServoId = 0 (H) or 1 (V) 
// ServoPosition = Percentage : 0% = left, 50% = center, 100% = right
/////////////////////////////////////////////////////////////////////////

int setServo(int ServoId, int ServoPosition)
{
  
  if(servoDev==NULL) return(-1);
 
 // ajustement for vertical
 if (ServoId == SERVO_V)
 {
 	ServoPosition +=15;
 	if (ServoPosition>100) ServoPosition = 100;
 }
 // if we need to reverse Servo
  if (b_reverseServo[ServoId]) ServoPosition = 100-ServoPosition;
  
 // servo position is between 0 and 100%
 // we change the scale to be between g_servoMax_X and g_servoMin_X
  ServoPosition = g_servoMin[ServoId]+ 	(g_servoMax[ServoId]-g_servoMin[ServoId])*ServoPosition/100;
  
  // avoid extrem position for safety reason
 // if (ServoPosition < 5) ServoPosition = 5;
  //if (ServoPosition > 95) ServoPosition = 95;
  
  if (oldServoPosition[ServoId] != ServoPosition) // avoid unusefull move
  {
  	fprintf(servoDev, "%d=%d%%\n",ServoId,ServoPosition);
  	fflush(servoDev);
  	if (b_servoTrace==1) printf("(I) Move servo %d to %d%%\n",ServoId, ServoPosition);
  }
  // remember old position
  oldServoPosition[ServoId] = ServoPosition;
  return(1);
}

void centerServo(int ServoId)
{
	setServo(ServoId,50);
}

       
/////////////////////////////////////////////////////////////////////////
// compare Positions : return distance btw 2 points
/////////////////////////////////////////////////////////////////////////
int comparePositions(Point a,Point b)
{
	return (sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y)));
}


/////////////////////////////////////////////////////////////////////////
// moveEyesTo...
/////////////////////////////////////////////////////////////////////////
int moveEyesTo(VISITORS target)
{
	if (comparePositions(target.servo, previousServo) > g_nToleranceServo) // enough change to move eye
	{	
		if (nTrace==1) printf("(I) move servo H:%d V%d\n",target.servo.x,target.servo.y);
		previousServo = target.servo;
	
	
		int absX = imageMat.cols/2+(imageMat.cols/2*target.servo.x)/100;
		int absY = imageMat.rows/2-(imageMat.rows/2*target.servo.y)/100;
		
		//draw eye direction
		line(imageMat,Point(imageMat.cols/2,imageMat.rows/2),Point(absX,absY), Scalar( 0, 0, 255 ),2,8);
	
		// move servo
		setServo(SERVO_H,(target.servo.x+100)/2); // horizontal : change scale to 0-100
		setServo(SERVO_V,(target.servo.y+100)/2); // vertical : change scale to 0-100
	}
	
}

/////////////////////////////////////////////////////////////////////////
// Check if visitors are still here or temporarly left the screen
/////////////////////////////////////////////////////////////////////////
void checkIfExVisitorsDisappears( ) 
{	
	for (int j=0;j<g_nExVisitors;j++) 
    {
    	int bTmpFound = 0;
    	for (int i=0;i<g_nVisitors;i++)
    	{
    		// check if position are similar (not too far away)
    		int d = comparePositions(currentVisitors[i].pos,exVisitors[j].pos);
    		if (d < currentVisitors[i].size)
    		{
    			// it's close
    			bTmpFound = 1;
    			exVisitors[j].off = 0; // is back, number of frames off = 0
    		}
    	}		         	
    	
    	if (bTmpFound == 0) // no current visitor is close to this ex visitors
    	{
    		exVisitors[j].off++;
    		if (nTrace==1) printf("(I) %d left temporarly since %d frame\n",j,exVisitors[j].off);
    	}
    	
    }
}
     
/////////////////////////////////////////////////////////////////////////
// Remove old visitors = these where off = MaxFameOff
/////////////////////////////////////////////////////////////////////////
void removeOldVisitors( )
{
	int isOff[MAX_VISITOR];
	int i,j,nOff=0;
	
	// init 0
	for (i = 0; i<MAX_VISITOR;i++) isOff[i] = 0;
	
	// for each ex visitor, check is number of frame off is reached
	for (i=0;i<g_nExVisitors;i++)
	{
		// if too many frames without him
		if (exVisitors[i].off >= g_nMaxFrameOff)
		{
			nOff++;
			isOff[i] = 1;		
		}
	}
	for (i=0;i<g_nExVisitors;i++)
	{
		// if limit is reached
		if (isOff[i] == 1)
		{
			for (j=i;j<g_nExVisitors;j++)
			{
				exVisitors[j] = exVisitors[j+1];
			}
		}
	}
	g_nExVisitors = g_nExVisitors - nOff;
	
	if ((nOff>0)&&(nTrace==1)) printf("(I) %d visitors disappears\n",nOff);
	
}

/////////////////////////////////////////////////////////////////////////
// initialization stuff...
/////////////////////////////////////////////////////////////////////////

int initSequence( )
{
	// init random stuff
	srand(time(NULL));

	// init sequence : load face recognition model 
	// load face model
    if (!face_cascade.load("/usr/share/opencv/lbpcascades/lbpcascade_frontalface.xml"))
    // to slow //if (!face_cascade.load("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"))
   	{
    			cout <<"(E) face cascade model not loaded\n"; 
    			return -1;
    }
    trace("(I) Load Face modele : ok\n");
    
/*	if (!eye_cascade.load("/usr/share/opencv/haarcascades/haarcascade_eye.xml"))
    {
    			cout <<"(E) eye cascade model not loaded\n"; 
    			return -1;
    }
    trace("(I) Load Eye model : ok\n");*/
	trace("(I) Traces activated\n");   
	if (nTrace==1) printf("(I) number of frames max before a visitor disappears=%d\n",  g_nMaxFrameOff);
	
		// initialize camera
    capture = raspiCamCvCreateCameraCapture(0); // Index doesn't really matter
	if (capture)
	{
		 trace("(I) Camera initialization : ok\n");
	}
	else { trace ("(E) camera can't be initialized\n"); return -1; }
	
	// init data
	for (int i=0;i<MAX_VISITOR;i++) { exVisitors[i].pos.x = 0; exVisitors[i].pos.y=0; exVisitors[i].off = 0;}
	g_nExVisitors 	= 0;
	
	// previous servoMove = 0
	previousServo.x =0; previousServo.y= 0; 
	
	// last looked visitor
	lastLookedVisitor.off 	= 0;
	lastLookedVisitor.pos 	= Point(0,0);
	lastLookedVisitor.servo = Point(0,0);
	lastLookedVisitor.size	= 0;
	
	// no visitor
	noVisitor.off 	= 0;
	noVisitor.pos 	= Point(0,0);
	noVisitor.size	= 0;
	noVisitor.servo = Point(0,0);
	
	
	// init servoblaster
   servoDev=fopen("/dev/servoblaster","w");
   if (servoDev==NULL)
   {
   		trace("(E) ServoBlaster init failed (unable to open /dev/servoblaster)\n"); 
   		return -1;
   }
	// center servo
	
	oldServoPosition[SERVO_H] = 0;
  	oldServoPosition[SERVO_V] = 0;
  	
	centerServo(SERVO_H);
	centerServo(SERVO_V);
	
	// last time blink eye
	lastBlinkTime 	= time(NULL);
	lastNapTime		= time(NULL);
	
	// openEye
	if (b_reverseServo[SERVO_B]!=2) setServo(SERVO_B,100);
	
	return 0;
}

////////////////////////////////////////////////
// blink eye strategy
////////////////////////////////////////////////
int checkBlinkEye (void)
{
	double now = time(NULL);
	double interval;
	
	if ((now-lastBlinkTime+rand()%6-3)>g_blinkPeriod) // add a +/- 3 seconds random
	{
		trace("(I) Blink eye (periodic)\n");
		
		interval = g_halfBlinkDuration/100;
		
		// blink eye
		for (int i=100;i>=0;i--)
		{
			setServo(SERVO_B,i);
			usleep(interval*1000);	// micro seconds
		}
		for (int i=0;i<=100;i++)
		{
			setServo(SERVO_B,i);
			usleep(interval*1000);	// micro seconds
		}
		lastBlinkTime = now;
	}
	return  1;
}

////////////////////////////////////////////////
// blink eye strategy
////////////////////////////////////////////////
int checkNapTime(void)
{
	double now = time(NULL);
	double interval = 3*g_halfBlinkDuration/100;
		
	if ((now-lastNapTime+rand()%(int)(g_napPeriod*0.2) - g_napPeriod*0.1)>g_napPeriod) // add a +/- 10% random
	{
		trace("(I) Time to a nap ! \n");
		// it's time for a nap !
		b_napOngoing = 1;
		
		// close slowly eye
		for (int i=100;i>=0;i--)
		{
			setServo(SERVO_B,i);
			usleep(interval*1000);	// micro seconds
		}
		// wait
		sleep(g_napDuration);
		
		
		// i'm ready. Open eye
		for (int i=0;i<=100;i++)
		{
			setServo(SERVO_B,i);
			usleep(interval*1000);	// micro seconds
		}
		trace("(I) Nap finished ! \n");
		lastNapTime = now;
	}
	b_napOngoing = 0;
	return 1;
}

//////////////////////////////////////////////////////////////////////////
// Move Eye Circle : like "i'm boring !"
//////////////////////////////////////////////////////////////////////////
void eyeCircle(int sens)
{
	double alpha;
	int x,y;
	trace("(I) I'm boring ! \n");
	
	for (int i=0;i<36;i++)
	{
		if (sens>0) alpha = 31.0/180.0*(float)(i);
		else alpha = -31.0/180.0*(float)(i);
		
		x = 50+(int)(50.0*cos(alpha));
		y = 50+(int)(50.0*sin(alpha));
		
		setServo(SERVO_H,x);
		setServo(SERVO_V,y);	
		
		usleep(30*1000);	// micro seconds	
	}
	sleep(1);
	// RAZ
	setServo(SERVO_V,50);		
	setServo(SERVO_H,50);		
		
}

//////////////////////////////////////////////////////////////////////////
// Move Eye Circle : like "where are you ? I'm alone !"
//////////////////////////////////////////////////////////////////////////
void eyeSearch(int cycle, int direction)
{
	
	trace("(I) Where are you ? \n");
	
	if (direction == SERVO_H)
	{
		setServo(SERVO_V,50);		
		for (int i=0;i<cycle;i++)
		{
			setServo(SERVO_H,0);
			sleep(3);
			setServo(SERVO_H,100);
			sleep(3);	
		}
	}
	else
	{
		setServo(SERVO_H,50);		
		for (int i=0;i<cycle;i++)
		{
			setServo(SERVO_V,0);
			sleep(3);
			setServo(SERVO_V,100);
			sleep(3);	
		}
	}
	//RAZ
	setServo(SERVO_V,50);		
	setServo(SERVO_H,50);		

}

//////////////////////////////////////////////////////////////////////////
//
// Main loop
//
/////////////////////////////////////////////////////////////////////////

int main(int argc, const char** argv)
{
	// variables declaration
	vector< Rect_<int> > faces, eyes; // rectangle for faces and eyes
	cv::Mat faceDetected ;			// faces detected
	int im_width ;					// width image
   	int im_height;					// height image
	int posEyeX, posEyeY;			// position of first eye
	int posCenterX, posCenterY;		// position of center of image
	char sPos[20];
		// number of ex visitors
	
		// check syntax	
 	if (argc != 20)
 	{
	       printf("usage : %s trace(0/1/2)  ToleranceFramesOff (5=2s) distMinToMoveEye(20) nbFramesMaxLooking@sameguy (5) MaxServo_X (95) MinServo_X (5) MaxServo_Y (95) MinServo_Y (5) reverseServo_H (0) reverseServo_V (0) reverseServo_B (0) MaxServo_B (95) MinServo_B (5) blinkPeriod (30) halfBlinkDuration (1000) napPeriod (600) napDuration(30) servoTrace(0)\n", argv[0]);
	       printf("- ToleranceFramesOff = number of frames without the guy before considering the visitor left\n");
	       printf("- distMinToMoveEye = max distance of a guy' move (in pixels) to be ignored by servo (avoid micro move) \n");
	       printf("- nbFramesMaxLooking@sameguy = number of frames on the same visitor before changing \n");
	       printf("- MaxServo_H = Max Amplitude for Servo Horizontal (in percent)\n");
	       printf("- MinServo_H = Min Amplitude for Servo Horizontal (in percent)\n");
	       printf("- MaxServo_V = Max Amplitude for Servo Vertical (in percent)\n");
	       printf("- MinServo_V = Min Amplitude for Servo Vertical (in percent)\n");
	       printf("- ReverseServo_H : 0=No , 1=Yes\n");
	       printf("- ReverseServo_V : 0=No , 1=Yes\n");
	       printf("- ReverseServo_B : 0=No , 1=Yes 2 = NO BLINK SERVO\n");
	       printf("- MaxServo_B = Max Amplitude for Servo Blink (in percent) =eye open\n");
	       printf("- MinServo_B = Min Amplitude for Servo Blink (in percent)=eye close\n");
	       printf("- BinkPeriod = time btw 2 blink eyes (in seconds)\n");
	       printf("- HalfBlinkDuration = time to close eye (milliseconds)\n");
	       printf("- napPeriod = time between 2 naps (seconds)\n");
	       printf("- napDuration = duration of a nap (seconds)\n");
	       printf("- servoTrace = display trace for all servos movement (0=no/1=yes)\n");
	       printf("- behaviorFreq = frequence of special eyes behavior (0-1000) 10 = once per minute)\n");
	       exit(1);
	}
	
	nTrace 					= atoi(argv[1]);	// 0 = nothing 1 = text+video 2 = video
	g_nMaxFrameOff			= atoi(argv[2]);	// number of frames accepted before seeing the visitor left		
	g_nToleranceServo 		= atoi(argv[3]);	// min distance to move eye (avoid micro change)
    g_nMax_Frames_Looked 	= atoi(argv[4]);	// number of frames before changing guy to look
    
    g_servoMax[SERVO_H]		 = atoi(argv[5]);	// max amplitude of servo H (in percent)
    g_servoMin[SERVO_H]		 = atoi(argv[6]); // min amplitude of servo H (in percent)
    g_servoMax[SERVO_V]		 = atoi(argv[7]); // max amplitude of servo V (in percent)
    g_servoMin[SERVO_V]		 = atoi(argv[8]); // min amplitude of servo V (in percent)
 	b_reverseServo[SERVO_H]  = atoi(argv[9]);	// reverse signal of servo H
	b_reverseServo[SERVO_V]  = atoi(argv[10]);	// reverse signal of servo V
	b_reverseServo[SERVO_B]  = atoi(argv[11]);	// no reverse signal of servo B
	g_servoMax[SERVO_B]		 = atoi(argv[12]);  // max amplitude of servo Blinking (in percent)
    g_servoMin[SERVO_B]		 = atoi(argv[13]);  // min amplitude of servo Blinking (in percent)
	g_blinkPeriod			 = atoi(argv[14]);  // blink period
	g_halfBlinkDuration		 = atoi(argv[15]);  // time to close eye
	g_napPeriod				 = atoi(argv[16]);  // time to close eye
	g_napDuration			 = atoi(argv[17]);  // time to close eye
	b_servoTrace			 = atoi(argv[18]);  // 0 = no trace for all servo movement, 1 = trace
	g_nFreqBehavior			 = atoi(argv[19]);  // freq of behavoir : 0-1000 (=100%)
	b_napOngoing			 = 0;	// no nap ongoing
	trace("-------------------------\n");
	trace("(C) Pierre RAUFAST - 2014\n");
	trace("-------------------------\n");

	if (nTrace==1) printf("(I) Max/Min Servo Horizontal =(%d%% / %d%%)\n",  g_servoMax[SERVO_H],g_servoMin[SERVO_H]);
	if (nTrace==1) printf("(I) Max/Min Servo Vertical =(%d%% / %d%%)\n",  g_servoMax[SERVO_V],g_servoMin[SERVO_V]);
	if (nTrace==1) printf("(I) Max/Min Servo Blink =(%d%% / %d%%)\n",  g_servoMax[SERVO_B],g_servoMin[SERVO_B]);
	if ((nTrace==1)&&(b_reverseServo[SERVO_H]==1)) printf("(I) Servo Horizontal is in REVERSE mode\n");
    if ((nTrace==1)&&(b_reverseServo[SERVO_V]==1)) printf("(I) Servo Vertical is in REVERSE mode\n");
    if ((nTrace==1)&&(b_reverseServo[SERVO_B]==1)) printf("(I) Servo Blink is in REVERSE mode\n");
    if ((nTrace==1)&&(b_reverseServo[SERVO_B]==2)) printf("(I) No Servo Blink : Blink Eyes and Nap disactivated\n");
   
    if (initSequence()<0) return -1;
     
	
		// create a display window 
	if (nTrace)	cvNamedWindow("WhatPappSees NO1", CV_WINDOW_AUTOSIZE);
	do {
		
		// get the picture from webcam
		IplImage* image = raspiCamCvQueryFrame(capture);
		imageMat =cvarrToMat(image); 
			
		// draw center of webcam = position  
		im_width = imageMat.cols; im_height = imageMat.rows;
	  	posCenterX = im_width/2; posCenterY = im_height/2;
		if (nTrace) line(imageMat,Point(posCenterX,posCenterY-10),Point(posCenterX,posCenterY+10), Scalar( 0, 255, 0 ),1,8);
	    if (nTrace) line(imageMat,Point(posCenterX-10,posCenterY),Point(posCenterX+10,posCenterY), Scalar( 0, 255, 0 ),1,8);
	    
	    //
		// detect face and analyse each visitors
		//
		//face_cascade.detectMultiScale(imageMat, faces, 1.1, 3, CV_HAAR_SCALE_IMAGE, Size(160,160)); // detect face
		face_cascade.detectMultiScale(imageMat, faces, 1.1, 3,CV_HAAR_SCALE_IMAGE,Size(80,80)); // detect face
		g_nVisitors = faces.size();	 		// number of visitors found
		if (g_nVisitors>MAX_VISITOR) g_nVisitors = MAX_VISITOR; // limit to 10
		
	
		for (int i=0;i<g_nVisitors;i++)
		{
			Rect faceRect = faces[i]; // rectangle around the face
			
			// assumption : eye = center of the face
		 	posEyeX = (faceRect.tl().x+faceRect.br().x)/2 ;
           	posEyeY = (faceRect.tl().y+faceRect.br().y)/2 ;
           	
           	// remember positions of visitors
           	currentVisitors[i].pos.x = posEyeX;
           	currentVisitors[i].pos.y = posEyeY;
            currentVisitors[i].size  = faceRect.width/2;
              	 
            // compute position of move if needed
           	currentVisitors[i].servo.x = ((posEyeX-posCenterX)*100)/(im_width/2);
           	currentVisitors[i].servo.y = -((posEyeY-posCenterY)*100)/(im_height/2);
           	
           	// trace
           	sprintf(sPos,"%d (H=%d V=%d)",i,currentVisitors[i].servo.x,currentVisitors[i].servo.y);
           	if (nTrace) putText(imageMat, sPos, Point(posEyeX, posEyeY), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1.0);
        	
        	// draw circle = distance max to recognize a move or a new guy
        	circle(imageMat, currentVisitors[i].pos, currentVisitors[i].size,Scalar( 255,0,0),1,8);
          
		}
		
		 
		 // now, check if an old visitors has disappeared = out of the screen while too many frames or just temporarly
         checkIfExVisitorsDisappears();
         
         // and remove old visitors if needed
         removeOldVisitors();
         
         // remember current visitor
         for (int i=0;i<g_nVisitors;i++)
         {
         	int bMatch = 0;
         	// check if an ex visitors match
         	for (int j=0;j<g_nExVisitors;j++)
         	{
         		int d = comparePositions(currentVisitors[i].pos,exVisitors[j].pos);
         		if (d < currentVisitors[i].size)
         		{
         			// it matchs : remember new position
         			bMatch = 1;
         			exVisitors[j] = currentVisitors[i];
    	      	}
         	}
         	// if nobody match, it's a new visitor	
         	if (bMatch == 0)
         	{
         		// add new comer
         		exVisitors[g_nExVisitors] = currentVisitors[i];
         		exVisitors[g_nExVisitors].off = 0;
         		g_nExVisitors++;
         		lastLookedVisitor = currentVisitors[i];
         		b_newVisitors = 1;
         		if (nTrace==1) printf("(I) %d is a new visitor\n",i);
         	}
         	else b_newVisitors = 0;	// no new visitor
         } 
         
           
	////////////////////////////////////////////////////////////
	// strategy to move eyes
	////////////////////////////////////////////////////////////


	// if at least 1 visitor ...
	if (g_nExVisitors>0)
	{
		if (g_nExVisitors == 1) // only one visitor, just follow him
		{
			
			moveEyesTo(currentVisitors[0]);
		}
		else // many visitors
		{
			// if no new visitor
			if (!b_newVisitors) 
			{
				int distMin 	= 32000;
				int distMax 	= 0;
				int d;
				int nWhoMin		= 0;
				int nWhoMax		= 0;
				int nWhoRnd		= 0;
				for (int i=0;i<g_nExVisitors;i++)
				{
					// get the visitor the closest 
				 	//d = comparePositions(currentVisitors[i].pos,lastLookedVisitor.pos);
         			d = comparePositions(exVisitors[i].pos,lastLookedVisitor.pos);
         		
         			if (d<distMin)
         			{
         				distMin = d;
         				nWhoMin = i;
         			}
         			if (d>=distMax)
         			{
         				distMax = d;
         				nWhoMax = i;
         			}
				}
				nWhoRnd = rand()%g_nExVisitors;
				
				// check number of times this guy is looked
				d = comparePositions(exVisitors[nWhoMin].pos,lastLookedVisitor.pos);
         		if (d<exVisitors[nWhoMin].size) lastLookedVisitor.off++;	
         		
         		// if looked too long times
         		if (lastLookedVisitor.off >= g_nMax_Frames_Looked)
         		{
         			int nWho;
         			// select the farthest
         			if (nWhoRnd!=nWhoMin) nWho = nWhoRnd; else nWho = nWhoMax;
         			
         			lastLookedVisitor.pos = exVisitors[nWho].pos;
         			lastLookedVisitor.servo = exVisitors[nWho].servo;
         			lastLookedVisitor.off = 0;
         			trace("(I) I look somebody else\n");
         		}
         		else
         		{
         			//select the closest
         			lastLookedVisitor.pos = exVisitors[nWhoMin].pos;
         			lastLookedVisitor.servo = exVisitors[nWhoMin].servo;
         
         		}
				
			}
			// move eyes
			moveEyesTo(lastLookedVisitor);
		}
		
	}
	else
	{
		// nobody : look in the middle
		moveEyesTo(noVisitor);
		
		// random behavior
		int h = rand()%1000;
		if (h<g_nFreqBehavior)
		{
			// select randomly a behavior
			switch(rand()%4) 
			{
				case 0: eyeCircle(1); break;
				case 1: eyeCircle(-1); break;
				case 2: eyeSearch(2,SERVO_H);break;
				case 3: eyeSearch(2,SERVO_V);break;
			}
		}
		
	}
	
	/////////////////////////////////////////////////////////////
	// Blink eye strategy
	/////////////////////////////////////////////////////////////
	if (b_reverseServo[SERVO_B]!=2)
	{
		if (b_napOngoing==0) checkBlinkEye(); // check if we need to blink eye
		checkNapTime(); // check if it's time to nap
	}
	
	// show what the painting sees
	if (nTrace) imshow("WhatPappSees NO2", imageMat);
		
    } while (cvWaitKey(10) < 0); // until key is pressed

	// for test
	/*eyeCircle(1); // draw circle with eyes
	sleep(1);
	eyeCircle(-1); // draw circle with eyes
	sleep(1);
	eyeSearch(1,SERVO_H);
	eyeSearch(1,SERVO_V);
	*/	
	// center eye beofre leaving
	centerServo(SERVO_H);
	centerServo(SERVO_V);

	// close all stuff properly
	fclose(servoDev);
        
	if (nTrace) cvDestroyWindow("WhatPappSees NO3");
	raspiCamCvReleaseCapture(&capture);
	return 0;
}
