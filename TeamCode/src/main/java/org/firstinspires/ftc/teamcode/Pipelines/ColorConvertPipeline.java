package org.firstinspires.ftc.teamcode.Pipelines;

import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

/***
 * This pipeline shows how to do different color conversions to different spaces to see what happens and to play with them.
 */

public class ColorConvertPipeline extends OpenCvPipeline {
    //required to send telemetry.
    Telemetry telemetry;
    public ColorConvertPipeline(Telemetry telemetry) {
        this.telemetry = telemetry;
    }
    //end required to send telemetry.


    Mat R = new Mat();
    Mat G = new Mat();
    Mat B = new Mat();
    Mat GRAY = new Mat();
    Mat HSV = new Mat();
    Mat YCrCb = new Mat();
    Mat Y = new Mat();
    Mat Cr = new Mat();
    Mat Cb = new Mat();

    @Override
    public void init(Mat input) {
        // Executed before the first call to processFrame for subframes and other presistent frames.
        //inputGray = input; //needed?
        //inputHSV = input;
        //inputHSV_Range = input;
    }

    @Override
    public Mat processFrame(Mat input) {
        // Executed every time a new frame is dispatched
        Imgproc.cvtColor(input,GRAY,Imgproc.COLOR_RGB2GRAY);
        Imgproc.cvtColor(input,HSV,Imgproc.COLOR_RGB2HSV);
        Imgproc.cvtColor(input,YCrCb,Imgproc.COLOR_RGB2YCrCb);
        Core.extractChannel(YCrCb,Y,0);
        Core.extractChannel(YCrCb,Cr,1);
        Core.extractChannel(YCrCb,Cb,2);
        Core.extractChannel(input,R,0);
        Core.extractChannel(input,G,1);
        Core.extractChannel(input,B,2);

        telemetry.addData("Mean R Value: ", Core.mean(R));
        telemetry.addData("Mean G Value: ", Core.mean(G));
        telemetry.addData("Mean B Value: ", Core.mean(B));
        telemetry.addData("Mean Y Value: ", Core.mean(Y));
        telemetry.addData("Mean Cr Value: ", Core.mean(Cr));
        telemetry.addData("Mean Cb Value: ", Core.mean(Cb));
        telemetry.update();
        //return input;   // Return the image that will be displayed in the viewport
        // uncomment out the image you want to see.
        return YCrCb;
        //return Y;
        //return Cr;
        //return Cb;
        //return R;
        //return G;
        //return B;
        // (In this case the input mat directly)
    }


    @Override
    public void onViewportTapped() {
        // Executed when the image display is clicked by the mouse or tapped
        // This method is executed from the UI thread, so be careful to not
        // perform any sort heavy processing here! Your app might hang otherwise
    }

}