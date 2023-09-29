package org.firstinspires.ftc.teamcode.Pipelines;

import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

/***
 * This pipeline is just for showing the basic structure of a pipeline
 */

public class BlankPipeline extends OpenCvPipeline {
    Telemetry telemetry;
    public BlankPipeline(Telemetry telemetry) {
        this.telemetry = telemetry;
    }

    //BOUNDS for inRange
    static final Scalar BLUE = new Scalar(0, 0, 255);
    static final Scalar GREEN_LOWER_BOUND = new Scalar(45, 7, 0);
    static final Scalar GREEN_UPPER_BOUND = new Scalar(68, 255, 218);
    static final Scalar PURPLE_LOWER_BOUND = new Scalar(124, 13, 54);
    static final Scalar PURPLE_UPPER_BOUND = new Scalar(39, 153, 255);
    static final Scalar YELLOW_LOWER_BOUND = new Scalar(14, 88, 120);
    static final Scalar YELLOW_UPPER_BOUND = new Scalar(31, 255, 255);
    //establishing top left points of rectangle and the x and y difference
    //to the bottom right point
    //resoltion of image = 4032 x 3024
    static final Point REGION1_TOPLEFT_ANCHOR_POINT = new Point(100, 200);
    static final Point REGION2_TOPLEFT_ANCHOR_POINT = new Point(275, 200);
    static final Point REGION3_TOPLEFT_ANCHOR_POINT = new Point(425, 200);
    static final int REGION_WIDTH = 100;
    static final int REGION_HEIGHT = 100;



    //Reestablishing the points (A) which will be the anchor points for
    //points B

    Point region1_pointA = new Point(
            REGION1_TOPLEFT_ANCHOR_POINT.x, REGION1_TOPLEFT_ANCHOR_POINT.y);
    Point region1_pointB = new Point(
            REGION1_TOPLEFT_ANCHOR_POINT.x + REGION_WIDTH,
            REGION1_TOPLEFT_ANCHOR_POINT.y + REGION_HEIGHT);
    Point region2_pointA = new Point(
            REGION2_TOPLEFT_ANCHOR_POINT.x, REGION2_TOPLEFT_ANCHOR_POINT.y);
    Point region2_pointB = new Point(
            REGION2_TOPLEFT_ANCHOR_POINT.x + REGION_WIDTH,
            REGION2_TOPLEFT_ANCHOR_POINT.y + REGION_HEIGHT);
    Point region3_pointA = new Point(
            REGION3_TOPLEFT_ANCHOR_POINT.x, REGION3_TOPLEFT_ANCHOR_POINT.y);
    Point region3_pointB = new Point(
            REGION3_TOPLEFT_ANCHOR_POINT.x + REGION_WIDTH,
            REGION3_TOPLEFT_ANCHOR_POINT.y + REGION_HEIGHT);

    //Variables for Mats and Avgs
    Mat region1_HSV, region2_HSV, region3_HSV;
    Mat HSV = new Mat();
    Mat GreenBinaryMat = new Mat();
    Mat PurpleBinaryMat = new Mat();
    Mat YellowBinaryMat = new Mat();
    Mat maskedGreenInputMat = new Mat();
    Mat maskedPurpleInputMat = new Mat();
    Mat maskedYellowInputMat = new Mat();
    int hAvgLeft, hAvgMiddle, hAvgRight;
    int sAvgLeft, sAvgMiddle, sAvgRight;
    int vAvgLeft, vAvgMiddle, vAvgRight;

    void inputToHSV(Mat input){
        Imgproc.cvtColor(input, HSV, Imgproc.COLOR_RGB2HSV);
    }
    @Override
    public void init(Mat firstFrame) {
        //to initialize Cb object for submats to be linked
        inputToHSV(firstFrame);
        region1_HSV = HSV.submat(new Rect(region1_pointA, region1_pointB));
        region2_HSV = HSV.submat(new Rect(region2_pointA, region2_pointB));
        region3_HSV = HSV.submat(new Rect(region3_pointA, region3_pointB));
    }

    @Override
    public Mat processFrame(Mat input) {
        inputToHSV(input);
        //Values to output avg values withing regions to telemetry
        hAvgLeft = (int) Core.mean(region1_HSV).val[0];
        sAvgLeft = (int) Core.mean(region1_HSV).val[1];
        vAvgLeft = (int) Core.mean(region1_HSV).val[2];
        hAvgMiddle = (int) Core.mean(region2_HSV).val[0];
        sAvgMiddle = (int) Core.mean(region2_HSV).val[1];
        vAvgMiddle = (int) Core.mean(region2_HSV).val[2];
        hAvgRight = (int) Core.mean(region3_HSV).val[0];
        sAvgRight = (int) Core.mean(region3_HSV).val[1];
        vAvgRight = (int) Core.mean(region3_HSV).val[2];
        //cont.
        telemetry.addData("hAvgLeft", hAvgLeft);
        telemetry.addData("sAvgLeft", sAvgLeft);
        telemetry.addData("vAvgLeft", vAvgLeft);
        telemetry.addData("hAvgMiddle", hAvgMiddle);
        telemetry.addData("sAvgMiddle", sAvgMiddle);
        telemetry.addData("vAvgMiddle", vAvgMiddle);
        telemetry.addData("hAvgRight", hAvgRight);
        telemetry.addData("sAvgRight", sAvgRight);
        telemetry.addData("vAvgRight", vAvgRight);
        telemetry.update();

        //Right now, this is only able to check if green is on the left,
        //if purple is in the middle, and if yellow is on the right.
        //It cannot yet tell me where green is if it is not on the left, etc.
        Core.inRange(region1_HSV,GREEN_LOWER_BOUND, GREEN_UPPER_BOUND, GreenBinaryMat);
        //maskedInputMat.release();
        Core.inRange(region2_HSV,PURPLE_LOWER_BOUND, PURPLE_UPPER_BOUND, PurpleBinaryMat);
        //maskedInputMat.release();
        Core.inRange(region3_HSV, YELLOW_LOWER_BOUND, YELLOW_UPPER_BOUND, YellowBinaryMat);
        //maskedInputMat.release();




        if (45 < hAvgLeft && hAvgLeft < 68 && 7 < sAvgLeft && sAvgLeft < 255 && 0 < vAvgLeft && vAvgLeft < 218) {

        }
        else if (124 < hAvgMiddle && hAvgMiddle < 39 && 13 < sAvgMiddle && sAvgMiddle < 153 && 54 < vAvgMiddle && vAvgMiddle < 255) {

        } else if (14 < hAvgRight && hAvgRight < 31 && 88 < sAvgRight && sAvgRight < 255 && 120 < vAvgRight && vAvgRight < 255) {

        }
        /*
        We are going to make a bitwise mask for each color of pixel.

        */


        Imgproc.rectangle(
                HSV,
                region1_pointA,
                region1_pointB,
                BLUE,
                2);
        Imgproc.rectangle(
                HSV,
                region2_pointA,
                region2_pointB,
                BLUE,
                2);
        Imgproc.rectangle(
                HSV,
                region3_pointA,
                region3_pointB,
                BLUE,
                2);


        return HSV;
    }


    @Override
    public void onViewportTapped() {
        // Executed when the image display is clicked by the mouse or tapped
        // This method is executed from the UI thread, so be careful to not
        // perform any sort heavy processing here! Your app might hang otherwise
    }

}