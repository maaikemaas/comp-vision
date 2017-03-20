#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "Helper.h"
#include "Video.h"

using namespace cv;
using namespace std;

int main()
{



  // A class with helper functions
  Helper helper;
  
  // A class with for video input/output. In this case output to "output.avi" and input from device 0 (webcam)
  Video video("output.avi", 0);

  // Initialize the input (webcam)
  bool is_open_input = video.initializeInput();
  // Assert that it's really open
  CV_Assert(is_open_input);
  // Get the webcam caputure device
  auto video_capture = video.getCaptureDevice();
  // An image is a matrix (cv::Mat) in OpenCV
  Mat frame;
  // Request an image from the webcam
  while (frame.empty())
  	video_capture >> frame;

  // Some matrix properties of the image frame we just pulled from the webcam
  cout << "This is an image of type:        " << helper.showCVMatType(frame.type()) << endl;
  cout << "The pixel size is [WxH]:         " << frame.size() << endl;
  cout << "The image channel amount is:     " << frame.channels() << endl;
  cout << "The image pixel depth is:        " << frame.depth() << " [" << helper.showCVMatType(frame.depth()) << "]" << endl;
  cout << "The matrix size is [R,C]:        " << frame.rows << ", " << frame.cols << endl; 
  cout << "The matrix dimensions amount is: " << frame.dims << endl;

  // Open a Window
  namedWindow(WEBCAM_WINDOW, CV_WINDOW_FREERATIO);
  // Show the image we pulled from the webcam in the named window
  imshow(WEBCAM_WINDOW, frame);

  // Write a message to the console
  cout << "Select the Webcam window and press a key to continue..." << endl << endl;
  // Wait for any keyboard input
  waitKey();
  // Remove the opened window
  destroyWindow(WEBCAM_WINDOW);
  // Open a new window, different name
  namedWindow(IMAGE_WINDOW, CV_WINDOW_FREERATIO);

  /*
   A deep copy means that every value in the matrix is duplicated. This is in contrast to 
   a shallow copy, where only the memory address of the data is copied. A deep copy is slow, 
   but changing values in the deep copied matrix will not change the values in the original.   
  */
  // Clone (deep copy) the webcam frame image
  Mat img_matrix = frame.clone();
  // Shallow copy the webcam image (try it, find out the difference!)
  //Mat img_matrix = frame;

  /*
  Create a vector of unique (integer) random numbers. This way is better than using a RNG directly,
  because we avoid duplicate numbers
  */
  helper.initRandomIntVector(min<int>(img_matrix.rows, img_matrix.cols));
  auto random_numbers = helper.getRandomIntVector();

  // Assign some random values for pixel locations, taken from vector positions 0 and 1 of the random numbers vector
  int x0 = random_numbers[0];
  int y0 = random_numbers[1];

  // Note that a color in an image is a 3 value vector, due to convention, the color order in the vector is: B G R (instead of RGB)
  cout << "The color vector of pixel [" << x0 << ", " << y0 << "] is: " << img_matrix.at<Vec3b>(y0, x0) << endl;
  cout << "Let's paint a red circle at that location" << endl;

  // Also paint on some text to go with the red dot
  std::stringstream text;
  text << "[" << x0 << ", " << y0 <<"]";
  // Paint the text with nice black outline
  Helper::putPrettyText(text.str(), Point(x0, y0), 0.8, img_matrix);
  // Paint the colored dot
  circle(img_matrix, Point(x0, y0), 3, Color_RED_LIGHT, -1, CV_AA);

  // Of course now the color value of the R channel at the chosen location is the maximal value (255)
  const int red_channel = 2;
  cout << "The color value of pixel [" << x0 << ", " << y0 << "] at the RED color channel is: " << (int) img_matrix.at<Vec3b>(y0, x0)[red_channel] << endl;

  imshow(IMAGE_WINDOW, img_matrix);
  cout << "Select the Image window and press a key to continue..." << endl << endl;
  waitKey();

  // Convert the 3 channel color image to a 1 channel gray image
  Mat gray_img_matrix;
  cvtColor(frame, gray_img_matrix, COLOR_BGR2GRAY);

  // New random location, taken from vector positions 3 and 4 of the random numbers vector
  int x1 = random_numbers[3];
  int y1 = random_numbers[4];

  // Gray image properties
  cout << "The gray image type is: " << helper.showCVMatType(gray_img_matrix.type()) << endl;
  cout << "The gray image channel amount is: " << gray_img_matrix.channels() << endl;

  // Note no vector anymore, just 1 value (cast to integer, otherwise it will show as an ASCII character because its type is unsigned char)
  cout << "The gray value of pixel [" << x1 << ", " << y1 << "] is: " << (int) gray_img_matrix.at<uchar>(y1, x1) << endl;

  imshow(IMAGE_WINDOW, gray_img_matrix);
  cout << "Select the Image window and press a key to continue..." << endl << endl;
  waitKey();

  // Commonly 32 bit (float) images with 1 channel have real values between [0 .. 1]
  // To convert to that, we need to scale 255 to 1
  // imshow will properly show unscaled CV_32F too though (I think... try it!)
  Mat gray_32bit_image;
  gray_img_matrix.convertTo(gray_32bit_image, CV_32F, 1 / 255.0);

  // Gray 32F image properties
  cout << "The gray float image type is: " << helper.showCVMatType(gray_32bit_image.type()) << endl;
  cout << "The gray float image channel amount is: " << gray_32bit_image.channels() << endl;

  // New random locations
  int x2 = random_numbers[5];
  int y2 = random_numbers[6];

  // Note no vector anymore, just 1 value (cast to integer, otherwise it will show as an ASCII character)
  cout << "The gray value of pixel [" << x2 << ", " << y2 << "] is: " << gray_32bit_image.at<float>(y2, x2) << endl;

  // Let's concatenate the color image and the gray image horizontally.
  // You can figure out yourself why we have to use cvtColor here!
  Mat gray_as_color_matrix;
  cvtColor(gray_img_matrix, gray_as_color_matrix, COLOR_GRAY2BGR);

  // Concatenate horizontally (stick right to left, must have same height!)
  hconcat(img_matrix, gray_as_color_matrix, img_matrix);

  // Some common matrix types
  cout << "A 3 channel 8 bit (color image) matrix reports as " << helper.showCVMatType(DataType<Vec3b>::type) << endl;
  cout << "A 1 channel 8 bit (gray image) matrix reports as " << helper.showCVMatType(DataType<uchar>::type) << endl;
  cout << "A matrix of type int (1 channel 16 bit) reports as " << helper.showCVMatType(DataType<int>::type) << endl;
  cout << "A matrix of type float (1 channel 32 bit) reports as " << helper.showCVMatType(DataType<float>::type) << endl;
  cout << "A matrix of type double (1 channel 64 bit) reports as " << helper.showCVMatType(DataType<double>::type) << endl;

  // Allocate a matrix
  Mat matrix(4, 4, CV_32F);
  // Garbage values are shown, because we only allocated the matrix, we did not initialize it!
  cout << "Unallocated garbage stuff: " << matrix << endl;
  // Allocate a matrix and initialize with zeros
  Mat zero_matrix = Mat::zeros(2, 4, CV_32F);
  // Now it's defined with zeros (slower)
  cout << "Zeros: " << zero_matrix << endl;
  // Allocate a matrix and initialize with ones
  Mat ones_matrix = Mat::ones(4, 2, CV_32S);
  // Now it's defined with ones (equally slower)
  cout << "Ones: " << ones_matrix << endl;
  // Allocate a matrix and initialize with a random value
  Mat random_matrix = Mat(3, 5, CV_64F, Scalar::all(sqrt(random_numbers[7])));
  // Now it's defined with the given value (even slower than cv::Mat::zeros or cv::Mat::ones)
  cout << "One random value: " << random_matrix << endl;
  // Define a random vector of integers
  vector<int> random_vector(random_numbers.begin() + 8, random_numbers.begin() + 8 + (3 * 4 * 5));
  // Shallow copy the vector to a Mat. We're not copying values, just pointing the memory of the matrix to the vector
  Mat more_random_matrix = Mat(random_vector);
  cout << "All random value vector: " << more_random_matrix.t() << endl;
  cout << "The type is: " << helper.showCVMatType(more_random_matrix.type()) << endl;

  int value = more_random_matrix.at<int>(4, 0);
  random_vector[4] = -42;
  // See? We didn't touch the matrix, but it's value has changed!
  cout << "Implicitly changed value: " << value << " to: " << more_random_matrix.at<int>(4, 0) << endl;

  // Create an image from the random vector
  Mat more_random_image;
  more_random_matrix.convertTo(more_random_image, CV_8U);
  // Normalize it to fit the color space [0 ... 255 ]
  normalize(more_random_image, more_random_image, 255, 0, NORM_MINMAX);

  // The vector length is divisible by 3 and by 4, so we can reshape it into a color image!
  // Reshape the vector to a 4x5 matrix with 3 color channels
  more_random_image = more_random_image.reshape(3, 4); // This is fast, because it doesn't copy any values!

  // Upscale the image to the size of img_matrix
  Mat scaled_mrm;
  resize(more_random_image, scaled_mrm, img_matrix.size());
  // Stick it to the bottom of img_matrix (must have same width!)
  vconcat(img_matrix, scaled_mrm, img_matrix);

  cout << "We will save this beautiful image to a file. The image extension decides the type of image." << endl;
  imwrite("image.jpg", img_matrix);

  cout << "Select the Image window and press a key to continue..." << endl << endl;
  imshow(IMAGE_WINDOW, img_matrix);
  waitKey();
  destroyWindow(IMAGE_WINDOW);
  
  /*
  In this next part we'll write a video
  This may not work right away, because you may not have any good codecs installed!
  Make sure you have a codec pack, such as the K-lite codec pack: http://www.codecguide.com/download_k-lite_codec_pack_mega.htm

  To initialize the video output, we need to give it the size (WxH) of the frames we will write to it. 
  This size cannot change after inintializing.
  */
  bool is_open_output = video.initializeOutput(frame.size());
  CV_Assert(is_open_output);
  auto video_writer = video.getWriterDevice();

  cout << "A video is a sequence of images. Which means you keep reading images from the webcam in" << endl;
  cout << "a loop with a small delay to catch pressed keys (1 ms)." << endl;
  cout << "We will try to record the sequence and write it to an AVI video file called output.avi" << endl;
  cout << "Press the <ESC> key to stop the loop and quit." << endl;

  // Location for some writing on the frame
  Point base_location(8, 24);

  // Time measures
  int64 t, t0 = getTickCount();
  int64 counter = 0;

  // Keyboard input
  int key = -1;
  
  // Open a Window
  namedWindow(WEBCAM_WINDOW, CV_WINDOW_FREERATIO);

  /*
   * We create a track bar with an integer trackbar_value connected to it by reference.
   * This means that when trackbar_value is updated by the trackbar, its value
   * will also change here.
   */
  int trackbar_value = 0;
  createTrackbar(TRACKBAR_NAME, WEBCAM_WINDOW, &trackbar_value, 99);

  // As long as key is not <ESC> loop
  while (key != 27)
  {
    // Retrieve a frame from the webcam
    video_capture >> frame;
    
    // Use the track bar value to scale the video image
    if (trackbar_value != 0)
    {
      double scale = 1 / (double)(trackbar_value + 1);
      resize(frame, frame, Size(), scale, scale);
      // Scale back to original size, now using INTER_NEAREST as interpolation to
      // create a block effect
      resize(frame, frame, Size(), 1 / scale, 1 / scale, INTER_NEAREST);
    }

    // Flip the image horizontally to get intuitive movement
    flip(frame, frame, 1);

    // Calculate time running and FPS
    t = getTickCount();
    double time_spent = (t - t0) / getTickFrequency();
    double fps = counter++ / time_spent;

    // Write on the frame
    std::stringstream text;
    text << cvRound(time_spent) << "s [" << cvRound(fps) << "fps]";

    // Print the frame rate into the image
    Helper::putPrettyText(text.str(), base_location, 0.8, frame);

    // Write the frame to the video file
    video_writer << frame;

    // Show the frame image
    imshow(WEBCAM_WINDOW, frame);

    // Get the keyboard input and wait 10ms to allow timely writing
    key = waitKey(10);
  }

  // Release the video_writer (finish writing)
  video_writer.release();

  // Remove all open windows
  destroyAllWindows();

  // Return error code 0 (no errors) to the console
  return EXIT_SUCCESS;
}
