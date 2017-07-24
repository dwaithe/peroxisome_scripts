
title = getTitle();
dir = getDirectory("image");
setBatchMode(true);
roiManager("Reset");
run("Clear Results");
run("Set Scale...", "distance=0 known=0 pixel=1 unit=pixel");
run("Set Measurements...", "area mean standard modal min centroid center perimeter bounding shape feret's integrated median skewness kurtosis redirect=None decimal=9");
run("Colors...", "foreground=white background=black selection=yellow");

rand = false;
if (rand == true){
roiManager('open',dir+title+'-randROIset.zip');	}else{
roiManager('open',dir+title+'-ROIset.zip');}


//setBatchMode(true);
//roiManager("Select",0);
//run("Duplicate...", "title=add_to");
//run("32-bit");
roiCount = roiManager("Count");
count = 0;
patch_sum = 0;
print("\\Clear");
print(title+"\tgfp_int\tgfp_int_norm\tmean_int_norm\tsum_of_gray_norm\tpixel_area_norm\tcirc_norm\tperimeter_norm\tmean_int\tsum_of_gray\tpixel_area\tcirc\tperimeter\tmean_int_norm_2\tsum_of_gray_norm_2\tpixel_area_norm_2\tcirc_norm_2\tperimeter_norm_2\tmean_int_2\tsum_of_gray_2\tpixel_area_2\tcirc_2\tperimeter_2\tM1_arr\tM2_arr\tpear\tpear_flip");

max_mean = 0;
max_sum_of_gray = 0;
max_pixel_area = 0;
max_circ = 0;
max_perimeter =0;
mean_arr = newArray(roiCount);
sum_of_gray_arr = newArray(roiCount);
pixel_area_arr = newArray(roiCount);
circ_arr = newArray(roiCount);
perimeter_arr = newArray(roiCount);

gfp_mean = 0;
max_mean_2 = 0;
max_sum_of_gray_2 = 0;
max_pixel_area_2 = 0;
max_circ_2 = 0;
max_perimeter_2 =0;
gfp_arr = newArray(roiCount);
mean_arr_2 = newArray(roiCount);
sum_of_gray_arr_2 = newArray(roiCount);
pixel_area_arr_2 = newArray(roiCount);
circ_arr_2 = newArray(roiCount);
perimeter_arr_2 = newArray(roiCount);
M1_arr = newArray(roiCount);
M2_arr = newArray(roiCount);
pear = newArray(roiCount);
pear_flip = newArray(roiCount);
flip = true;
//roiCount = 10;
for (i=0;i<roiCount;i++){
	
	
	selectWindow(title);
	setSlice(1);
	roiManager("Select",i);
	run("Measure");
	gfp_arr[i] = getResult("Mean",nResults-1);
	pear[i] = calc_pearsons(title, i,false);
	
	if (flip == true){
		pear_flip[i] = calc_pearsons(title, i,true);
		
		}
	
	
	run("Duplicate...", "title=patch_"+i);
	run("32-bit");
	//run("Enhance Contrast...", "saturated=0.8");
	
	//run("Select All");
	
	//y = getProfile();
	
	//setKeyDown("alt");
	//y1 = getProfile();
	//setKeyDown("none");
	//x = newArray(y.length);

		
	
	//for(b=0;b<y.length;b++){
	//x[b] = b;}
	
	//Fit.doFit("Gaussian", x, y);
	//r2 = Fit.rSquared;
	//Fit.doFit("Gaussian", x, y1);
	//r2b = Fit.rSquared;
	//print("a="+r2+"b="+r2b);
	
	//if (r2 > 0.90 && r2b >0.90){
		//print("keep");
		//close();
		selectWindow(title);
		setSlice(2);
		run("Duplicate...", "title=patch_raw_ch0_"+i);
		selectWindow(title);
		setSlice(2);
		run("Duplicate...", "title=patch_ch0_"+i);
		run("Clear Outside");
		
		run("Measure");
		mean_arr[i] = getResult("Mean",nResults-1);
		sum_of_gray_arr[i] = getResult("RawIntDen",nResults-1);
		
		
		run("8-bit");
		setAutoThreshold("RenyiEntropy dark");
		run("Convert to Mask");
		run("Despeckle");
		run("Erode");
		run("Create Selection");
		
		if (selectionType() != -1){
		
		run("Measure");
		
		pixel_area_arr[i] = getResult("RawIntDen",nResults-1)/255;
		circ_arr[i] = getResult("Circ.",nResults-1);
		perimeter_arr[i] = getResult("Perim.",nResults-1);}
		else{
		pixel_area_arr[i] = 0;
		circ_arr[i] = 0;
		perimeter_arr[i] = 0;
		}
		
		

		selectWindow(title);
		setSlice(1);
		run("Duplicate...", "title=patch_raw_chB_"+i);
		selectWindow(title);
		setSlice(4);
		run("Duplicate...", "title=patch_raw_ch1_"+i);
		selectWindow(title);
		setSlice(4);
		run("Duplicate...", "title=patch_ch1_"+i);
		run("Clear Outside");
		
		run("Measure");
		mean_arr_2[i] = getResult("Mean",nResults-1);
		sum_of_gray_arr_2[i] = getResult("RawIntDen",nResults-1);
		
		
		run("8-bit");
		setAutoThreshold("RenyiEntropy dark");
		run("Convert to Mask");
		run("Despeckle");
		run("Erode");
		run("Create Selection");
		
		if (selectionType() != -1){
		//run("Make Inverse");
		run("Measure");
		
		pixel_area_arr_2[i] = getResult("RawIntDen",nResults-1)/255;
		circ_arr_2[i] = getResult("Circ.",nResults-1);
		perimeter_arr_2[i] = getResult("Perim.",nResults-1);}
		else{
		pixel_area_arr_2[i] = 0;
		circ_arr_2[i] = 0;
		perimeter_arr_2[i] = 0;
		}
		
		count = count +1;

		imageCalculator("AND create 32-bit", "patch_ch0_"+i, "patch_ch1_"+i);
		run("Measure");
		close();
		and_arr = getResult("RawIntDen",nResults-1)/255;

		M1_arr[i] = and_arr/pixel_area_arr[i];
		M2_arr[i] = and_arr/pixel_area_arr_2[i];
		
		run("Merge Channels...", " c1=patch_ch1_"+i+" c2=patch_ch0_"+i+"   create ignore");
		selectWindow("Composite");
		run("RGB Color");
		selectWindow("Composite (RGB)");
		rename("patch_col");
		selectWindow("Composite");
		close();
		run("Merge Channels...", "c1=patch_raw_ch1_"+i+" c2=patch_raw_ch0_"+i+" c3=patch_raw_chB_"+i+" create ignore");
		selectWindow("Composite");
		run("RGB Color");
		selectWindow("Composite (RGB)");
		rename("patch_raw");
		selectWindow("Composite");
		close();
		//selectWindow("patch_ch1_"+i);
		//close();
		
		
	if (max_mean <mean_arr[i]){
		max_mean = mean_arr[i];
		}
	if (max_sum_of_gray <sum_of_gray_arr[i]){
		max_sum_of_gray = sum_of_gray_arr[i];
		}
	if (max_pixel_area <pixel_area_arr[i]){
		max_pixel_area =pixel_area_arr[i];
		}
	if (max_circ < circ_arr[i]){
		max_circ = circ_arr[i];
		}
	if (max_perimeter < perimeter_arr[i]){
		max_perimeter = perimeter_arr[i];
		}
	
	
	if (gfp_mean <gfp_arr[i]){
		gfp_mean = gfp_arr[i];
		}
	
		
	
		

	if (max_mean_2 <mean_arr_2[i]){
		max_mean_2 = mean_arr_2[i];
		}
	if (max_sum_of_gray_2 <sum_of_gray_arr_2[i]){
		max_sum_of_gray_2 = sum_of_gray_arr_2[i];
		}
	if (max_pixel_area_2 <pixel_area_arr_2[i]){
		max_pixel_area_2 =pixel_area_arr_2[i];
		}
	if (max_circ_2 < circ_arr_2[i]){
		max_circ_2 = circ_arr_2[i];
		}
	if (max_perimeter_2 < perimeter_arr_2[i]){
		max_perimeter_2 = perimeter_arr_2[i];
		}
	
	
	
	
	}






	for (i=0;i<roiCount;i++){
		print("\t"+(gfp_arr[i])+"\t"+(gfp_arr[i]/gfp_mean)+"\t"+(mean_arr[i]/max_mean)+"\t"+(sum_of_gray_arr[i]/max_sum_of_gray)+"\t"+(pixel_area_arr[i]/max_pixel_area)+"\t"+(circ_arr[i]/max_circ)+"\t"+(perimeter_arr[i]/max_perimeter)+"\t"+(mean_arr[i])+"\t"+(sum_of_gray_arr[i])+"\t"+(pixel_area_arr[i])+"\t"+(circ_arr[i])+"\t"+(perimeter_arr[i])+"\t"+(mean_arr_2[i]/max_mean_2)+"\t"+(sum_of_gray_arr_2[i]/max_sum_of_gray_2)+"\t"+(pixel_area_arr_2[i]/max_pixel_area_2)+"\t"+(circ_arr_2[i]/max_circ_2)+"\t"+(perimeter_arr_2[i]/max_perimeter_2)+"\t"+(mean_arr_2[i])+"\t"+(sum_of_gray_arr_2[i])+"\t"+(pixel_area_arr_2[i])+"\t"+(circ_arr_2[i])+"\t"+(perimeter_arr_2[i])+"\t"+(M1_arr[i])+"\t"+(M2_arr[i])+"\t"+pear[i]+"\t"+pear_flip[i]);
		
		
		
		}

selectWindow(title);
close();
//selectWindow("Results");
//rename(title+"-results.txt");
selectWindow("Log");  //select Log-window 
if (rand==true){
run("Text...", "save=["+dir+title+"-rand_results.txt]");}else{
run("Text...", "save=["+dir+title+"-results.txt]");
}
//saveAs("results", dir+title+"-results.txt"); 
run("Images to Stack", "method=[Copy (center)] name=patch_col title=patch_col use");
//run("Invert LUT");
run("Make Montage...", "increment=1 border=1 font=12");
if (rand==true){
saveAs("png", dir+title+"-rand_output.png");}else{
saveAs("png", dir+title+"-output.png");
}
//saveAs("results", dir+title+"-results.txt"); 
run("Images to Stack", "method=[Copy (center)] name=patch_raw title=patch_raw use");
//run("Invert LUT");
run("Make Montage...", "increment=1 border=1 font=12");
if (rand==true){
saveAs("png", dir+title+"-gray_rand_output.png");
}else{
saveAs("png", dir+title+"-gray_output.png");
}

//run("Z Project...", "projection=[Sum Slices]");
rename("output"+title);
function calc_pearsons(img1, roi_index,flip) {
//Calculates the Pearson product-moment correlation coefficient for two images.
//Will measure pearson's in an image region if a ROI is present.
//Make sure we are in patch mode for speed.

//setBatchMode(false);
//Select first input image.
selectWindow(img1);
setSlice(2);
//Duplicate this image
run("Duplicate...", "title="+img1+"-1");
//Set to floating point number because we want to do math on the image
run("32-bit");
roiManager("Select",roi_index);
//Measure in the image.
run("Measure");
//Save the mean and stdev
mean1 = getResult("Mean",nResults-1);
std1 = getResult("StdDev",nResults-1);
//Subtract the mean.
run("Subtract...", "value="+mean1);

//Do the same for the second image.
selectWindow(img1);
setSlice(4);
run("Duplicate...", "title="+img1+"-2");
run("32-bit");
if (flip== true){
	run("Flip Horizontally");
	
	}
run("32-bit");
roiManager("Select",roi_index);
run("Measure");
mean2= getResult("Mean",nResults-1);
std2 = getResult("StdDev",nResults-1);
run("Subtract...", "value="+mean2);

//Multiply the pixels of the two image.
imageCalculator("Multiply create 32-bit", img1+"-1", img1+"-2");
//Measure the correct region.
roiManager("Select",roi_index);
run("Measure");
rename(img1+"-meas");
//The mean of the normalised dot product.
mdot = getResult("Mean",nResults-1);
//Clean up a few things.
selectWindow(img1+"-1");
close();
selectWindow(img1+"-2");
close();
selectWindow(img1+"-meas");
close();

//Return the calculation
//setBatchMode(true);
return mdot/(std1*std2)
}


//setBatchMode(false);
//close();
//run("Enhance Contrast...", "saturated=0.8");
//rename("output"+title);
