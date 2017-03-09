This project explores the problem of automated taxonomification for a specific domain
of images, namely Vespidae wasp images. As of 2011, state of the art machine learning
and computer vision methodologies are leveraged for automated classification.

This project was sponsored by the PRIME (Pacific Rim Undergraduate Research
Experiences) Program and the Taiwan Forest Research Institute in Taipei, Taiwan. 

This project is no longer supported. 

Created by Adrian Teng-Amnuay.


Usage:
Compile the program and run from command line via "program.exe" <mode> <directory>
<mode> is 0 for training, 1 for testing.
<directory> if training mode is specified, this directory contains images grouped
by subfolder, resembling trainimages/A/1.jpg, trainimages/A/2.jpg, trainimages/B/1.jpg, etc.
if testing mode is specified, this directory contains only images to be labeled.


Configuration and setup:
Please ensure that you have OpenCV 2.3 and Python 2.7 installed

Please ensure that wasp images are all facing to the right
    If not, you can select them right click and select "Rotate clockwise"

Please ensure that wasp images have been compressed to ~ 20kb each
    You can do this in Windows by right clicking an image, open in Microsoft Office 
    Picture manager.
    Hit ctrl+t to view in thumbnails
    Select all
    Then go to Picture -> Compress pictures
    On the right hand side, select Compress for: Web pages
    Hit ok
    Then save all, and your pictures are compressed

The following files need to be in the same directory:
- program.exe 
- get_paths.py
- Folder for training or Folder for testing (or both)
    Training folder contains subfolders (for each species) with images in each
    Testing folder contains just the images
