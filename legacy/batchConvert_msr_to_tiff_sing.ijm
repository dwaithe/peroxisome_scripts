title = getInfo("image.filename");
dir =getDirectory("image");
getVoxelSize(Vox_w, Vox_h, Vox_d, unit);
//xtransAPD2 =0;// (1.00/0.08)*Vox_w;
//ytransAPD2 =0;// (-0.500/0.08)*Vox_h;
xtransAPD2 = (2.9/0.02)*Vox_w;
ytransAPD2 = (-1.7/0.02)*Vox_h;


run("Close All");
run("Bio-Formats Importer", "open=["+dir+title+"] autoscale color_mode=Default  view=Hyperstack stack_order=XYCZT series_1 ");
rename("APD1");
run("32-bit");
run("Bio-Formats Importer", "open=["+dir+title+"] autoscale color_mode=Default  view=Hyperstack stack_order=XYCZT series_2 ");
rename("APD2");
run("32-bit");
run("TransformJ Translate", "x-translation="+(xtransAPD2 )+" y-translation="+(ytransAPD2)+" z-translation=0.0 interpolation=Linear background=0.0");
selectWindow("APD2");
run("Close");
selectWindow("APD2 translated");
rename("APD2");

//run("Bio-Formats Importer", "open=["+dir+title+"] autoscale color_mode=Default  view=Hyperstack stack_order=XYCZT series_3 ");
//run("Bio-Formats Importer", "open=["+dir+title+"] autoscale color_mode=Default  view=Hyperstack stack_order=XYCZT series_4 ");

run("Merge Channels...", "c1=[APD1] c2=[APD2]  create");
selectWindow("Composite");
setSlice(1);
run("Blue");
setSlice(2);
run("Green");
//run("16-bit");
