package org.firstinspires.ftc.teamcode.Pipelines;

import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.openftc.easyopencv.OpenCvPipeline;

import java.util.ArrayList;

/***
 * This example shows how to use an OpenCvPipeline to process an image to draw a contour around an object.  It works by doing the following steps.
 * 0. Establish variables to hold onto a series of images (Mats) at different stages of analysis.
 * 1. Establish Scalar variables to establish upper and lower values for a range mask
 * 2. Make erode and dilate elements for the erosion and dilation of an image (to blur it to remove artifacts).
 *
 * PROCESSFRAME
 * 3. Make an ArrayList to hold onto all contours
 * 4. Take input image and convert it into a different color space.  HSV color space is often easier for some colors.
 * 5. Make an image that is a mask of the converted image.  Each pixel is either 255 or 0 depending on if it is in range, isolating important parts from background.
 * 6. Erodes an image thereby reducing noise, but shrinks any edges down
 * 7. Dilates an image which blows the edges back up to the previous size.
 * 8. Finds contours around the mask.
 * 9. Copies the original input image to a new place to later draw on top of it for viewing.
 * 10. Calculates the "moments" of the contours and uses them to find the center of the object.
 * 11. Calculates a bounding box around the contours.
 * 12. Draws the bounding box and outputs data about where the center is.
 */

public class ContoursPixelLocatorRED extends OpenCvPipeline {

    //necessary bits for making telemetry work in a pipeline!
    Telemetry telemetry;
    public ContoursPixelLocatorRED(Telemetry telemetry) {
        this.telemetry = telemetry;
    }

    //imgs
    Mat clrConvertedMat = new Mat(); //mat after color conversion
    Mat thresholdMat1 = new Mat(); //mask -- y pixels are part of thing or n they are not
    Mat thresholdMat2 = new Mat();
    Mat morphedThreshold1 = new Mat();  //denoise
    Mat morphedThreshold2 = new Mat();
    Mat contoursOnPlainImageMat = new Mat(); //copy of original image to vandalize with contours

    //Boolean for team (Red = T, Blue = F)
    public Boolean teamColor = true;
    public enum ConeColor {RED, BLUE}
    ConeColor conecolor = ConeColor.RED;
    public enum ConePosition {LEFT,CENTER,RIGHT}
    ConePosition coneposition = ConePosition.CENTER;
    //upper and lower Scalar values for changing the range for the mask
    public Scalar lower1 = new Scalar(120,120,0);
    public Scalar upper1 = new Scalar(255,255,255);
    public Scalar lower2 = new Scalar(100,80,30);
    public Scalar upper2 = new Scalar(140,255,255);


    //sizes to adjust for the erosion and dilation of the mask
    public int size = 3;
    public int mult = 2;

    Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(size, size));
    Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(mult*size, mult*size));

    String centers = "";
    Point center1 = new Point(0,0); //buckets for the center of objects for later
    Point center2 = new Point(0, 0);
    @Override
    public Mat processFrame(Mat input) {
        teamColor = true; //reinit teamColor default Red
        // Executed every time a new frame is dispatched
        //Array to hold the contours
        ArrayList<MatOfPoint> contoursList = new ArrayList<>();
        ArrayList<Moments> momentsList = new ArrayList<>();
        ArrayList<Point> centers = new ArrayList<>();

        //convert color space to limit what you are seeing

        Imgproc.cvtColor(input, clrConvertedMat, Imgproc.COLOR_RGB2HSV);
        //input.copyTo(grayMat);
        //Make a mask using a range.  Includes only the values within that range
        Core.inRange(clrConvertedMat,lower1,upper1,thresholdMat1);
        Core.inRange(clrConvertedMat,lower2,upper2,thresholdMat2);
        //erode the image - reducing the noisy pixels from the mask
        //by taking the lowest value in a 3x3 box (erodeElement)
        Imgproc.erode(thresholdMat1, morphedThreshold1, erodeElement);
        Imgproc.erode(morphedThreshold1, morphedThreshold1, erodeElement);
        //For Second Object
        Imgproc.erode(thresholdMat2, morphedThreshold2, erodeElement);
        Imgproc.erode(morphedThreshold2, morphedThreshold2, erodeElement);
        //expands the edges of the mask out again (needed because erosion reduces both noise and signal
        //and this boosts the signal of the edges)
        Imgproc.dilate(morphedThreshold1, morphedThreshold1, dilateElement);
        Imgproc.dilate(morphedThreshold1, morphedThreshold1, dilateElement);
        //For Second Object
        Imgproc.dilate(morphedThreshold2, morphedThreshold2, dilateElement);
        Imgproc.dilate(morphedThreshold2, morphedThreshold2, dilateElement);
        //use function to find contours of each object and put them into contoursList.
        //find contours on the mask of all of the edges of the same value that make a closed shape.
        //These contours are points indicated on an image of their own, stored in contoursList.
        //Each object that is detected is stored as a separate object in contoursList.
        //We want to reduce this down to 1 object.

        //                  Mask Image        ListToStore  Optional   Mode                  Method
        //Mode and method are standard in example code.  Play around with other options
        Imgproc.findContours(morphedThreshold1,contoursList, new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_NONE);
        if(!contoursList.isEmpty()) {
            teamColor = true; //red
            conecolor = ConeColor.RED;
        }
        else {
            teamColor = false; //BLUE
            conecolor = ConeColor.BLUE;
        }
        Imgproc.findContours(morphedThreshold2,contoursList, new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_NONE);
        if(!contoursList.isEmpty() && !teamColor) {
            teamColor = false;
            conecolor = ConeColor.BLUE;
        }
        if (!contoursList.isEmpty()) {
            teamColor = true;
        }
        else {
            teamColor = false;
        }



        //copy "input" image to draw contours on for display purposes.  This just copies input to that place.
        input.copyTo(contoursOnPlainImageMat);

        //draw contours using the contours List
        //                  Image to draw on        contours2draw           ?       Color           Thickness
        Imgproc.drawContours(contoursOnPlainImageMat,contoursList,-1,new Scalar(0,0,255),2);

        //counter for the number of closed contours
        telemetry.addData("Number of objects", contoursList.size());
        //ArrayList<Moments> moments = new ArrayList<>();
        //ArrayList<Point> points = new ArrayList<>();

        Moments moment1 = new Moments();
        Moments moment2 = new Moments();
        center1 = new Point();


        //Sometimes images have no contours!  The list is empty, so if you try to grab an object from it
        //It yells at you.  This makes sure the list isn't empty before you access it!
        if (!contoursList.isEmpty()){
            //this grabs the first contour and finds the moments of it.

            for (MatOfPoint contour : contoursList){
                momentsList.add(Imgproc.moments(contour));  //make a moment for each contour and put it in the list
                centers.add(new Point(
                        (int)(momentsList.get(momentsList.size()-1).m10/momentsList.get(momentsList.size()-1).m00),
                        (int)(momentsList.get(momentsList.size()-1).m01/momentsList.get(momentsList.size()-1).m00))
                );  //calculates the centers and puts them into the centers list


            }


            moment1 = Imgproc.moments(contoursList.get(0));
            if (contoursList.size()>1) {
                moment2 = Imgproc.moments(contoursList.get(1));
            }
            //This calculations the center of mass using m10/m00, m01/m00 and stores them
            //as a point in "center"
            //the (int) is to cast the double to an integer to cut off the decimals...they were annoying
            center1 = new Point((int)(moment1.m10/moment1.m00),(int)(moment1.m01/moment1.m00));
            if (contoursList.size()>1) {
                center2 = new Point((int) (moment2.m10 / moment2.m00), (int) (moment2.m01 / moment2.m00));
            }


            //This draws a circle at that point on the image so we can see what the image tracks.
            Imgproc.circle(contoursOnPlainImageMat,center1,3,new Scalar(0,255,0),2);
            Imgproc.circle(contoursOnPlainImageMat,center2,3,new Scalar(0,255,0),2);
            //This calculates and draws a bounding rectangle.
            //Bounding rectangles are nice because they have an inherent width that can be measured
            Rect boundingRectangle1 = Imgproc.boundingRect(contoursList.get(0));
            Imgproc.rectangle(contoursOnPlainImageMat,boundingRectangle1.tl(),boundingRectangle1.br(),new Scalar(255,0,0),2);
            if (contoursList.size()>1) {
                Rect boundingRectangle2 = Imgproc.boundingRect(contoursList.get(1));
                Imgproc.rectangle(contoursOnPlainImageMat,boundingRectangle2.tl(),boundingRectangle2.br(),new Scalar(255,0,0), 2);
            }
            telemetry.addData("width of box ",boundingRectangle1.width);
            //At this point, we could try to estimate the distance of the box by a ratio of apparent size to expected.
            //The bigger the box width, the smaller the distance.
            //if we know a size of the box at a specific distance, if the box width doubles, then the distance halfs.

            //this draws a box around the image to show you where the computer things the object is based on "center"
        }
        //String centers = "Center 1: " + center1.x + ", " + center1.y + " Center 2: " + center2.x + ", " + center2.y;

        telemetry.addData("contour x location: ", center1.x);
        telemetry.addData("contour y location: ", center1.y);
        telemetry.addData("getAnalysis result (center point): ", center1);
        if (teamColor == true) {
            telemetry.addData("Team Color: ", "Red");
        }
        else {
            telemetry.addData("Team Color: ", "Blue");
        }
        telemetry.update();

        //this returns the image with the contours
        return contoursOnPlainImageMat; // Return the image that will be displayed in the viewport

        //uncomment these to see what they look like
        //return input;
        //return thresholdMat;
        //return morphedThreshold;

    }

    Point getCenter1() {
        return center1;
    }
    Point getCenter2() { return center2; }
    Mat getContoursOnPlainImageMat() {
        return contoursOnPlainImageMat;
    }
    public ConeColor getConeColor() {
        return conecolor;
    }
    Boolean getTeamColor() {
        return teamColor;
    }
    @Override
    public void onViewportTapped() {
        // Executed when the image display is clicked by the mouse or tapped
        // This method is executed from the UI thread, so be careful to not
        // perform any sort heavy processing here! Your app might hang otherwise
    }

}