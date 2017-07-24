//chromatic correction:
getVoxelSize(Vox_w, Vox_h, Vox_d, unit);
xtransAPD2 = (2.9/0.02)*Vox_w;
xtransAPD3 = (2.4/0.02)*Vox_w;
ytransAPD2 = (-1.7/0.02)*Vox_h;
ytransAPD3 = (-1.5/0.02)*Vox_h;




title = getInfo("image.filename");
dir =getDirectory("image");


//Open the image.
run("Close All");
run("Bio-Formats Importer", "open=["+dir+title+"] autoscale color_mode=Default  view=Hyperstack stack_order=XYCZT series_1 ");
rename("GFP_1");
run("32-bit");
run("Bio-Formats Importer", "open=["+dir+title+"] autoscale color_mode=Default  view=Hyperstack stack_order=XYCZT series_2 ");
rename("APD2");
run("32-bit");
run("TransformJ Translate", "x-translation="+(xtransAPD2 )+" y-translation="+(ytransAPD2)+" z-translation=0.0 interpolation=Linear background=0.0");
selectWindow("APD2");
run("Close");
selectWindow("APD2 translated");
rename("APD2");

run("Bio-Formats Importer", "open=["+dir+title+"] autoscale color_mode=Default  view=Hyperstack stack_order=XYCZT series_3 ");
rename("GFP_2");
run("32-bit");
run("Bio-Formats Importer", "open=["+dir+title+"] autoscale color_mode=Default  view=Hyperstack stack_order=XYCZT series_4 ");
rename("APD3");
run("32-bit");
run("TransformJ Translate", "x-translation="+(xtransAPD3 )+" y-translation="+(ytransAPD3)+" z-translation=0.0 interpolation=Linear background=0.0");
selectWindow("APD3");
run("Close");
selectWindow("APD3 translated");
rename("APD3");


//Find dimensions
width = getWidth();
height = getHeight();
//Specify search radius.
margin = 10;
//Set the output to zero. We keep the maximum value later.
max_value =0.0

//Make a copy of the original second channel.
selectWindow("GFP_2");
run("Duplicate...", "duplicate title=GFP_2_old");
//Crop the image to smaller area.
selectWindow("GFP_2");
run("Specify...", "width="+width-margin+" height="+height-margin+" x="+margin+" y="+margin+"");
run("Crop");

//Search neighbourhood.
for (i=0;i<18;i++){
	for(j=0;j<18;j++){

//Select the first window.
selectWindow("GFP_1");

xcorr =i;
ycorr =j;
roiManager("Reset");
//Add region to the manager.
run("Specify...", "width="+width-margin+" height="+height-margin+" x="+xcorr+" y="+ycorr+"");
roiManager("Add");

//Find the index of the submitted region.
roi_index1 = roiManager("Count")-1;
//Calculate product of the two regions with this function. (See end of script for function).
pea_out = calc_product("GFP_1","GFP_2",roi_index1);



if (pea_out > max_value){
	
	max_value = pea_out;
	i_out = i;
	j_out = j;
	}
}}



//Find the displacement 
idis =i_out-margin
jdis =j_out-margin


//Close the window which we cropped.
selectWindow("GFP_2");
close();
//Find the original.
selectWindow("GFP_2_old");
rename("GFP_2");

selectWindow("GFP_1");
run("TransformJ Translate", "x-translation="+(-idis)+" y-translation="+(-jdis)+" z-translation=0.0 interpolation=Linear background=0.0");

selectWindow("APD2");
run("TransformJ Translate", "x-translation="+(-idis)+" y-translation="+(-jdis)+" z-translation=0.0 interpolation=Linear background=0.0");

//Merge the channels.
run("Merge Channels...", "c1=[GFP_1 translated] c2=[APD2 translated] c3=GFP_2 c4=APD3 create");
selectWindow("Composite");
//Colour appropriately
setSlice(1);
run("Blue");
setSlice(2);
run("Green");
setSlice(3);
run("Magenta");
setSlice(4);
run("Red");



function calc_product(img1, img2,roi_index1) {
//Just in case running not in batchmode.
setBatchMode(true);
//Select first input image.
selectWindow(img1);
run("Select None");
//Duplicate this image
run("Duplicate...", "title="+img1+"-1");
//default region.
roiManager("Select",roi_index1);
run("Crop");
imageCalculator("Multiply create 32-bit", img1+"-1", img2);
//Measure the correct region.
run("Select All");
run("Measure");
//The mean of the normalised dot product.
mdot = getResult("Mean",nResults-1);
//Clean up a few things.
selectWindow(img1+"-1");
close();
selectWindow("Result of "+img1+"-1");
close();
return mdot
}
