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

public class SubmatPixelLocator extends OpenCvPipeline {
    Telemetry telemetry;

    public SubmatPixelLocator(Telemetry telemetry) {
        this.telemetry = telemetry;
    }

    //BOUNDS for inRange
    static final Scalar BLACK = new Scalar(0, 0, 0);
    static final Scalar YELLOW = new Scalar(255, 242, 0);
    static final Scalar PURPLE = new Scalar(128,0,128);
    static final Scalar GREEN = new Scalar(0,255, 0);
    static final Scalar GREEN_LOWER_BOUND = new Scalar(45, 7, 0);
    static final Scalar GREEN_UPPER_BOUND = new Scalar(68, 255, 218);
    static final Scalar PURPLE_LOWER_BOUND = new Scalar(89, 20, 51);
    static final Scalar PURPLE_UPPER_BOUND = new Scalar(143, 173, 255);
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
    Mat GreenRegion1BinaryMat = new Mat();
    Mat GreenRegion2BinaryMat = new Mat();
    Mat GreenRegion3BinaryMat = new Mat();
    Mat PurpleRegion1BinaryMat = new Mat();
    Mat PurpleRegion2BinaryMat = new Mat();
    Mat PurpleRegion3BinaryMat = new Mat();
    Mat YellowRegion1BinaryMat = new Mat();
    Mat YellowRegion2BinaryMat = new Mat();
    Mat YellowRegion3BinaryMat = new Mat();

    int AvgGreenRegion1, AvgGreenRegion2, AvgGreenRegion3, AvgPurpleRegion1, AvgPurpleRegion2, AvgPurpleRegion3,
            AvgYellowRegion1, AvgYellowRegion2, AvgYellowRegion3;


    public enum GreenPosition {
        LEFT, CENTER, RIGHT, UNKNOWN
    }

    public enum PurplePosition {
        LEFT, CENTER, RIGHT, UNKNOWN
    }

    public enum YellowPosition {
        LEFT, CENTER, RIGHT, UNKNOWN
    }

    private GreenPosition gPos = GreenPosition.LEFT;
    private PurplePosition pPos = PurplePosition.LEFT;
    private YellowPosition yPos = YellowPosition.LEFT;

    public enum Region1Pixel {
        GREEN, PURPLE, YELLOW, UNKNOWN
    }

    private enum Region2Pixel {
        GREEN, PURPLE, YELLOW, UNKNOWN
    }

    private enum Region3Pixel {
        GREEN, PURPLE, YELLOW, UNKNOWN
    }

    private Region1Pixel inRegion1 = Region1Pixel.GREEN;
    private Region2Pixel inRegion2 = Region2Pixel.GREEN;
    private Region3Pixel inRegion3 = Region3Pixel.GREEN;

    void inputToHSV(Mat input) {
        Imgproc.cvtColor(input, HSV, Imgproc.COLOR_RGB2HSV);
    }

    int maxOneTwo1, max1, maxOneTwo2, max2, maxOneTwo3, max3;

    @Override
    public void init(Mat firstFrame) {
        inputToHSV(firstFrame);
        region1_HSV = HSV.submat(new Rect(region1_pointA, region1_pointB));
        region2_HSV = HSV.submat(new Rect(region2_pointA, region2_pointB));
        region3_HSV = HSV.submat(new Rect(region3_pointA, region3_pointB));
    }

    @Override
    public Mat processFrame(Mat input) {
        inputToHSV(input);
        //Avgs of individual h/s/v values for each contour profile (we will have a contour for green
        //in all 3 regions, and a contour for purple in all 3 regions, etc, so we will need a total of
        //3 avgs per h/s/v value per region, so 9 avgs per region.

        Core.inRange(region1_HSV, GREEN_LOWER_BOUND, GREEN_UPPER_BOUND, GreenRegion1BinaryMat);
        Core.inRange(region2_HSV, GREEN_LOWER_BOUND, GREEN_UPPER_BOUND, GreenRegion2BinaryMat);
        Core.inRange(region3_HSV, GREEN_LOWER_BOUND, GREEN_UPPER_BOUND, GreenRegion3BinaryMat);
        Core.inRange(region1_HSV, PURPLE_LOWER_BOUND, PURPLE_UPPER_BOUND, PurpleRegion1BinaryMat);
        Core.inRange(region2_HSV, PURPLE_LOWER_BOUND, PURPLE_UPPER_BOUND, PurpleRegion2BinaryMat);
        Core.inRange(region3_HSV, PURPLE_LOWER_BOUND, PURPLE_UPPER_BOUND, PurpleRegion3BinaryMat);
        Core.inRange(region1_HSV, YELLOW_LOWER_BOUND, YELLOW_UPPER_BOUND, YellowRegion1BinaryMat);
        Core.inRange(region2_HSV, YELLOW_LOWER_BOUND, YELLOW_UPPER_BOUND, YellowRegion2BinaryMat);
        Core.inRange(region3_HSV, YELLOW_LOWER_BOUND, YELLOW_UPPER_BOUND, YellowRegion3BinaryMat);
        //Green
        AvgGreenRegion1 = (int) Core.mean(GreenRegion1BinaryMat).val[0];
        AvgGreenRegion2 = (int) Core.mean(GreenRegion2BinaryMat).val[0];
        AvgGreenRegion3 = (int) Core.mean(GreenRegion3BinaryMat).val[0];

        //Purple
        AvgPurpleRegion1 = (int) Core.mean(PurpleRegion1BinaryMat).val[0];
        AvgPurpleRegion2 = (int) Core.mean(PurpleRegion2BinaryMat).val[0];
        AvgPurpleRegion3 = (int) Core.mean(PurpleRegion3BinaryMat).val[0];

        //Yellow
        AvgYellowRegion1 = (int) Core.mean(YellowRegion1BinaryMat).val[0];
        AvgYellowRegion2 = (int) Core.mean(YellowRegion2BinaryMat).val[0];
        AvgYellowRegion3 = (int) Core.mean(YellowRegion3BinaryMat).val[0];

        //Checking max values for each region to determine location of pixels.
        //region 1
        //NOTE: LOGIC CURRENTLY HAS BLIND SPOTS NOTICED WHEN CHECKING images Pixel2, Pixel3 !
        //DOES NOT WORK WHEN GREEN AND PURPLE ARE IN THE "RIGHT" POSITION!
        maxOneTwo1 = Math.max(AvgGreenRegion1, AvgPurpleRegion1);
        max1 = Math.max(maxOneTwo1, AvgYellowRegion1);
        if (maxOneTwo1 == AvgGreenRegion1 && AvgGreenRegion1 > 10 && max1 == maxOneTwo1) {
            gPos = GreenPosition.LEFT;
            inRegion1 = Region1Pixel.GREEN;
        } else if (maxOneTwo1 == AvgPurpleRegion1 && AvgPurpleRegion1 > 10 && max1 == maxOneTwo1) {
            pPos = PurplePosition.LEFT;
            inRegion1 = Region1Pixel.PURPLE;
        } else if (max1 == AvgYellowRegion1 && AvgYellowRegion1 > 10) {
            yPos = YellowPosition.LEFT;
            inRegion1 = Region1Pixel.YELLOW;
        } else {
            inRegion1 = Region1Pixel.UNKNOWN;
        }

        //region 2
        maxOneTwo2 = Math.max(AvgGreenRegion2, AvgPurpleRegion2);
        max2 = Math.max(maxOneTwo2, AvgYellowRegion2);
        if (maxOneTwo2 == AvgGreenRegion2 && AvgGreenRegion2 > 10 && max2 == maxOneTwo2) {
            gPos = GreenPosition.CENTER;
            inRegion2 = Region2Pixel.GREEN;
        } else if (maxOneTwo2 == AvgPurpleRegion2 && AvgPurpleRegion2 > 10 && max2 == maxOneTwo2) {
            pPos = PurplePosition.CENTER;
            inRegion2 = Region2Pixel.PURPLE;
        } else if (max2 == AvgYellowRegion2 && AvgYellowRegion2 > 10) {
            yPos = YellowPosition.CENTER;
            inRegion2 = Region2Pixel.YELLOW;
        } else {
            inRegion2 = Region2Pixel.UNKNOWN;
        }
        //region 3
        maxOneTwo3 = Math.max(AvgGreenRegion3, AvgPurpleRegion3);
        max3 = Math.max(maxOneTwo3, AvgYellowRegion3);
        if (maxOneTwo3 == AvgGreenRegion3 && AvgGreenRegion3 > 10 && max3 == maxOneTwo3) {
            gPos = GreenPosition.RIGHT;
            inRegion3 = Region3Pixel.GREEN;
        } else if (maxOneTwo3 == AvgPurpleRegion3 && AvgPurpleRegion3 > 10 && max3 == maxOneTwo3) {
            pPos = PurplePosition.RIGHT;
            inRegion3 = Region3Pixel.PURPLE;
        } else if (max3 == AvgYellowRegion3 && AvgYellowRegion3 > 10) {
            yPos = YellowPosition.RIGHT;
            inRegion3 = Region3Pixel.YELLOW;
        } else {
            inRegion3 = Region3Pixel.UNKNOWN;
        }
        if (inRegion1 != Region1Pixel.GREEN && inRegion2 != Region2Pixel.GREEN && inRegion3 != Region3Pixel.GREEN) {
            gPos = GreenPosition.UNKNOWN;
        }
        if (inRegion1 != Region1Pixel.PURPLE && inRegion2 != Region2Pixel.PURPLE && inRegion3 != Region3Pixel.PURPLE) {
                pPos = PurplePosition.UNKNOWN;
        }
        if (inRegion1 != Region1Pixel.YELLOW && inRegion2 != Region2Pixel.YELLOW && inRegion3 != Region3Pixel.YELLOW) {
            yPos = YellowPosition.UNKNOWN;
        }
            //Green telemetry
            telemetry.addData("hAvgGreenRegion1", AvgGreenRegion1);
            telemetry.addData("hAvgGreenRegion2", AvgGreenRegion2);
            telemetry.addData("hAvgGreenRegion3", AvgGreenRegion3);

            //Purple telemetry
            telemetry.addData("hAvgPurpleRegion1", AvgPurpleRegion1);
            telemetry.addData("hAvgPurpleRegion2", AvgPurpleRegion2);
            telemetry.addData("hAvgPurpleRegion3", AvgPurpleRegion3);

            //Yellow telemetry
            telemetry.addData("hAvgYellowRegion1", AvgYellowRegion1);
            telemetry.addData("hAvgYellowRegion2", AvgYellowRegion2);
            telemetry.addData("hAvgYellowRegion3", AvgYellowRegion3);


            //Tells us which pixel is in each region
            telemetry.addData("LEFT PIXEL", inRegion1);
            telemetry.addData("CENTER PIXEL", inRegion2);
            telemetry.addData("RIGHT PIXEL", inRegion3);
            telemetry.addData("GREEN POS", gPos);
            telemetry.addData("PURPLE POS", pPos);
            telemetry.addData("YELLOW POS", yPos);


            telemetry.update();

            if (inRegion1 == Region1Pixel.GREEN) {
                Imgproc.rectangle(
                        HSV,
                        region1_pointA,
                        region1_pointB,
                        GREEN,
                        2);
            } else if (inRegion1 == Region1Pixel.PURPLE) {
                Imgproc.rectangle(
                        HSV,
                        region1_pointA,
                        region1_pointB,
                        PURPLE,
                        2);
            } else if (inRegion1 == Region1Pixel.YELLOW) {
                Imgproc.rectangle(
                        HSV,
                        region1_pointA,
                        region1_pointB,
                        YELLOW,
                        2);
            } else if (inRegion1 == Region1Pixel.UNKNOWN) {
                Imgproc.rectangle(
                        HSV,
                        region1_pointA,
                        region1_pointB,
                        BLACK,
                        2);
            } else {
                Imgproc.rectangle(
                        HSV,
                        region1_pointA,
                        region1_pointB,
                        BLACK,
                        2);
            }

            if (inRegion2 == Region2Pixel.GREEN) {
                Imgproc.rectangle(
                        HSV,
                        region2_pointA,
                        region2_pointB,
                        GREEN,
                        2);
            } else if (inRegion2 == Region2Pixel.PURPLE) {
                Imgproc.rectangle(
                        HSV,
                        region2_pointA,
                        region2_pointB,
                        PURPLE,
                        2);
            } else if (inRegion2 == Region2Pixel.YELLOW) {
                Imgproc.rectangle(
                        HSV,
                        region2_pointA,
                        region2_pointB,
                        YELLOW,
                        2);
            } else if (inRegion2 == Region2Pixel.UNKNOWN) {
                Imgproc.rectangle(
                        HSV,
                        region2_pointA,
                        region2_pointB,
                        BLACK,
                        2);
            } else {
                Imgproc.rectangle(
                        HSV,
                        region2_pointA,
                        region2_pointB,
                        BLACK,
                        2);
            }

            if (inRegion3 == Region3Pixel.GREEN) {
                Imgproc.rectangle(
                        HSV,
                        region3_pointA,
                        region3_pointB,
                        GREEN,
                        2);
            } else if (inRegion3 == Region3Pixel.PURPLE) {
                Imgproc.rectangle(
                        HSV,
                        region3_pointA,
                        region3_pointB,
                        PURPLE,
                        2);
            } else if (inRegion3 == Region3Pixel.YELLOW) {
                Imgproc.rectangle(
                        HSV,
                        region3_pointA,
                        region3_pointB,
                        YELLOW,
                        2);
            } else if (inRegion3 == Region3Pixel.UNKNOWN) {
                Imgproc.rectangle(
                        HSV,
                        region3_pointA,
                        region3_pointB,
                        BLACK,
                        2);
            } else {
                Imgproc.rectangle(
                        HSV,
                        region3_pointA,
                        region3_pointB,
                        BLACK,
                        2);
            }
        return HSV;
    }
    public void onViewportTapped() {
        // Executed when the image display is clicked by the mouse or tapped
        // This method is executed from the UI thread, so be careful to not
        // perform any sort heavy processing here! Your app might hang otherwise
    }
}

