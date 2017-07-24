//Macro written to find the foci in the first channel.
//Dominic Waithe (c) 2015.

//The first slice.
setSlice(1);

//Get the title of the file.
title = getTitle();

//Find the directory also.
dir = getDirectory("image");

//Duplicate the image.
run("Duplicate...","duplicate title=[copy]");

//Lightly blur the image.
run("Gaussian Blur...","sigma=2");
wid = getWidth();
hei = getHeight();

//Find the points based on the noise parameter (defualt for this application = 300).
run("Find Maxima...", "noise=10 output=[List]");
run("Find Maxima...", "noise=10 output=[Point Selection]");

//Select the image again.
selectWindow(title);
//Define array.
x_arr = newArray(nResults-1);
y_arr = newArray(nResults-1);

//Populate the array with results.
for(i=0;i<nResults-1;i++){
	x_arr[i] = getResult('X',i);
	y_arr[i] = getResult('Y',i);
	}

//Set the diameter of the region.
diameter = 20;
radius = diameter/2;

//Create an empty image of the same dimension as the input.
newImage("binary1", "8-bit black", getWidth(), getWidth(), 1);
roiManager("reset");

//We take all the points and then see if they are fully in the image or not. If they are we add them to the manager.
for(a=0;a<x_arr.length-1;a++){
start_x = x_arr[a];
start_y = y_arr[a];
if ((start_x - radius) > 0 && (start_y - radius)>0 && (start_x + radius) <wid &&(start_y + radius) <hei){
run("Specify...", "width="+diameter+" height="+diameter+" x="+start_x+" y="+start_y+" slice=1 oval centered");
roiManager("Add");}
}

//Take the regions and ink them in so we can test later whether the position has been taken.
roiManager("Select All");
roiManager("Fill");

//We save the ROI to a file.
roiManager('save',dir+title+'-ROIset.zip');
//Count the number of ROIs.
roiCount = roiManager("Count");

//We take all the points and then see if they are fully in the image or not. If they are we add them to the manager.
//Sets random points based around the existing points.
for (b=0;b<roiCount;b++){
	roiManager("Select",b);
	for (i=0;i<20;i++){
	//Generates random coordinates.
	x = (random()-0.5)*180;
	y = (random()-0.5)*180;
	Roi.getBounds(xpoint, ypoint, width, height);
	//Specifies the new location.
	run("Specify...", "width="+diameter+" height="+diameter+" x="+(xpoint+x)+" y="+(ypoint+y)+" slice=1 oval centered");
	run("Measure");
	mu = getResult("Mean",nResults-1);
	//If the region is not taken it will include it.
	if (mu ==0){
		roiManager("Add");
		i = 20;
		}}
	}
//Delete the ROI from before.
for (b=0;b<roiCount;b++){
	roiManager("Select",0);
	roiManager("Delete");
	}

//The roi manager save the roi.
roiManager('save',dir+title+'-randROIset.zip');
selectWindow(title);
close('//Others');
