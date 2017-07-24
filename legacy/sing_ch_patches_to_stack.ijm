title = getTitle();
dir = getDirectory("image");
roiManager("Reset");
run("Clear Results");

rand = false;

if (rand == true){
roiManager('open',dir+title+'-randROIset.zip');	}else{
roiManager('open',dir+title+'-ROIset.zip');}
run("Set Scale...", "distance=0 known=0 pixel=1 unit=pixel");
run("Set Measurements...", "area mean modal min centroid center perimeter bounding shape feret's integrated median skewness kurtosis redirect=None decimal=3");
run("Colors...", "foreground=white background=black selection=yellow");



setBatchMode(true);
roiCount = roiManager("Count");
count = 0;
patch_sum = 0;
print("\\Clear");
print(title+"\tgfp_int\tgfp_int_norm\tmean_int_norm\tsum_of_gray_norm\tpixel_area_norm\tcirc_norm\tperimeter_norm\tmean_int\tsum_of_gray\tpixel_area\tcirc\tperimeter");

gfp_mean = 0;
max_mean = 0;
max_sum_of_gray = 0;
max_pixel_area = 0;
max_circ = 0;
max_perimeter =0;
mean_arr = newArray(roiCount);
gfp_arr = newArray(roiCount);
sum_of_gray_arr = newArray(roiCount);
pixel_area_arr = newArray(roiCount);
circ_arr = newArray(roiCount);
perimeter_arr = newArray(roiCount);

for (i=0;i<roiCount;i++){
	
	
	selectWindow(title);
	setSlice(1);
	roiManager("Select",i);
	run("Measure");
	gfp_arr[i] = getResult("Mean",nResults-1);
	run("Duplicate...", "title=patch_raw_"+i);
	run("32-bit");
	//run("Enhance Contrast...", "saturated=0.8");
	
	run("Select All");
	
	//y = getProfile();
	
	//setKeyDown("alt");
	//y1 = getProfile();
	//setKeyDown("none");
	//x = newArray(y.length);

		
	
	///for(b=0;b<y.length;b++){
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
		run("Duplicate...", "title=patch_gray_"+i);
		selectWindow(title);
		setSlice(2);
		run("Duplicate...", "title=patch_thr_"+i);
		run("Clear Outside");
		
		run("Measure");
		mean_arr[i] = getResult("Mean",nResults-1);
		sum_of_gray_arr[i] = getResult("RawIntDen",nResults-1);
		
		//setAutoThreshold("RenyiEntropy dark");
		//setThreshold(5,255);
		run("8-bit");
		setAutoThreshold("MaxEntropy dark");
		setOption("BlackBackground",true);
		
		run("Convert to Mask");
		run("Despeckle");
		run("Erode");
		
		run("Create Selection");
		
		if (selectionType() != -1){
		//run("Make Inverse");
		run("Measure");
		
		pixel_area_arr[i] = getResult("RawIntDen",nResults-1)/255;
		circ_arr[i] = getResult("Circ.",nResults-1);
		perimeter_arr[i] = getResult("Perim.",nResults-1);}
		else{
		pixel_area_arr[i] = 0;
		circ_arr[i] = 0;
		perimeter_arr[i] = 0;
		}
		

		
		
		count = count +1;
		
		
	
		
	if (gfp_mean <gfp_arr[i]){
		gfp_mean = gfp_arr[i];
		}
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
	
	
	
	
	}

	for (i=0;i<roiCount;i++){
		print("\t"+(gfp_arr[i])+"\t"+(gfp_arr[i]/gfp_mean)+"\t"+(mean_arr[i]/max_mean)+"\t"+(sum_of_gray_arr[i]/max_sum_of_gray)+"\t"+(pixel_area_arr[i]/max_pixel_area)+"\t"+(circ_arr[i]/max_circ)+"\t"+(perimeter_arr[i]/max_perimeter)+"\t"+(mean_arr[i])+"\t"+(sum_of_gray_arr[i])+"\t"+(pixel_area_arr[i])+"\t"+(circ_arr[i])+"\t"+(perimeter_arr[i]));
		}

selectWindow(title);
close();


selectWindow("Log");  //select Log-window 
if (rand==true){
run("Text...", "save=["+dir+title+"-rand_results.txt]");}else{
run("Text...", "save=["+dir+title+"-results.txt]");
}

run("Images to Stack", "method=[Copy (center)] name=patch_thr title=patch_thr use");

run("Make Montage...", "increment=1 border=1 font=12");
if (rand==true){
saveAs("png", dir+title+"-rand_output.png");}else{
saveAs("png", dir+title+"-output.png");
}

run("Images to Stack", "method=[Copy (center)] name=patch_gray title=patch_gray use");

run("Make Montage...", "increment=1 border=1 font=12");
rename("montage1");
run("Images to Stack", "method=[Copy (center)] name=patch_raw title=patch_raw use");

run("Make Montage...", "increment=1 border=1 font=12");
rename("montage2");
run("Merge Channels...", "c2=montage1  c3=montage2 create ignore");


selectWindow("Composite");
run("RGB Color");
selectWindow("Composite (RGB)");
rename("patch_col");
selectWindow("Composite");
close();	


if (rand==true){
saveAs("png", dir+title+"-gray_rand_output.png");}else{
saveAs("png", dir+title+"-gray_output.png");
}


rename("output"+title);
//setBatchMode(false);

