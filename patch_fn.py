from sklearn.cluster import MeanShift, estimate_bandwidth
import PIL
from PIL import Image
from scipy.ndimage import filters
import csv
import pylab as P
import numpy as np
#collates that data.
def collate_data(data_store,cond_name_1, cond_name_2):
    cond1 = []
    cond2 = []
    for filename in data_store:
        try:
            if cond_name_1 in data_store[filename][cond_name_1]: pass
        except:print 'you have not measured this parameter or it has been misspelt or does not exist.',cond_name_1
        try:
            if cond_name_2 in data_store[filename][cond_name_2]: pass
        except:print 'you have not measured this parameter or it has been misspelt or does not exist.',cond_name_2
        cond1.extend(data_store[filename][cond_name_1])
        cond2.extend(data_store[filename][cond_name_2])
    return cond1,cond2

def calculate_measurements(data_store,diameter_of_roi,rand):
    ###Make the measurements for each file.
    for filename in data_store:
        if rand == False:
            xpts = data_store[filename]['o_xpts'] 
            ypts = data_store[filename]['o_ypts']
        else:
            xpts = data_store[filename]['rn_xpts'] 
            ypts = data_store[filename]['rn_ypts']

        ch0 = data_store[filename]['img_corr'][0,:,:]
        ch1 = data_store[filename]['img_corr'][1,:,:]
        ch2 = data_store[filename]['img_corr'][2,:,:]
        ch3 = data_store[filename]['img_corr'][3,:,:]

        radius = diameter_of_roi//2
        ym, xm = np.meshgrid(np.arange(-radius,radius+1),np.arange(-radius,radius+1))
        mask = np.sqrt(ym**2+xm**2) < radius

        ch0_ave = []
        ch1_ave = []
        ch2_ave = []
        ch3_ave = []
        ch0_sum = []
        ch1_sum = []
        ch2_sum = []
        ch3_sum = []

        ch13_pea_arr = []
        ch13_pea_flip_arr = []

        for xp, yp in zip(xpts,ypts):

            sq_reg0 = ch0[yp-radius:yp+radius+1,xp-radius:xp+radius+1]
            cir_reg0 = sq_reg0[mask]
            ch0_ave.append(np.average(cir_reg0))
            ch0_sum.append(np.sum(cir_reg0))

            sq_reg1 = ch1[yp-radius:yp+radius+1,xp-radius:xp+radius+1]
            cir_reg1 = sq_reg1[mask]
            ch1_ave.append(np.average(cir_reg1))
            ch1_sum.append(np.sum(cir_reg1))

            sq_reg2 = ch2[yp-radius:yp+radius+1,xp-radius:xp+radius+1]
            cir_reg2 = sq_reg2[mask]
            ch2_ave.append(np.average(cir_reg2))
            ch2_sum.append(np.sum(cir_reg2))

            sq_reg3 = ch3[yp-radius:yp+radius+1,xp-radius:xp+radius+1]
            cir_reg3 = sq_reg3[mask]
            ch3_ave.append(np.average(cir_reg3))
            ch3_sum.append(np.sum(cir_reg3))

            ch13_pea  = np.sum((np.array(cir_reg1)-np.average(cir_reg1))*(np.array(cir_reg3)-np.average(cir_reg3)))
            ch13_pea /= np.sqrt(np.sum((np.array(cir_reg1)-np.average(cir_reg1))**2))*np.sqrt(np.sum((np.array(cir_reg3)-np.average(cir_reg3))**2))  
            ch13_pea_arr.append(ch13_pea)


            cir_flip3 = np.fliplr(sq_reg3)[mask]
            ch13_pea_flip = np.sum((np.array(cir_reg1)-np.average(cir_reg1))*(np.array(cir_flip3)-np.average(cir_flip3)))
            ch13_pea_flip /= np.sqrt(np.sum((np.array(cir_reg1)-np.average(cir_reg1))**2))*np.sqrt(np.sum((np.array(cir_flip3)-np.average(cir_flip3))**2))  
            ch13_pea_flip_arr.append(ch13_pea_flip)


        ch0_ave_norm = np.copy(np.array(ch0_ave))/np.max(np.array(ch0_ave))
        ch1_ave_norm = np.copy(np.array(ch1_ave))/np.max(np.array(ch1_ave))
        ch2_ave_norm = np.copy(np.array(ch2_ave))/np.max(np.array(ch2_ave))
        ch3_ave_norm = np.copy(np.array(ch3_ave))/np.max(np.array(ch3_ave))

        if rand == False:
            data_store[filename]['ch0_ave'] = np.array(ch0_ave)
            data_store[filename]['ch1_ave'] = np.array(ch1_ave)
            data_store[filename]['ch2_ave'] = np.array(ch2_ave)
            data_store[filename]['ch3_ave'] = np.array(ch3_ave)

            data_store[filename]['ch0_sum'] = np.array(ch0_sum)
            data_store[filename]['ch1_sum'] = np.array(ch1_sum)
            data_store[filename]['ch2_sum'] = np.array(ch2_sum)
            data_store[filename]['ch3_sum'] = np.array(ch3_sum)

            data_store[filename]['ch0_ave_norm'] = np.array(ch0_ave_norm)
            data_store[filename]['ch1_ave_norm'] = np.array(ch1_ave_norm)
            data_store[filename]['ch2_ave_norm'] = np.array(ch2_ave_norm)
            data_store[filename]['ch3_ave_norm'] = np.array(ch3_ave_norm)

            data_store[filename]['ch13_pea'] = np.array(ch13_pea)
            data_store[filename]['ch13_pea_flip'] = np.array(ch13_pea_flip_arr)
        else:
            data_store[filename]['rand_ch0_ave'] = np.array(ch0_ave)
            data_store[filename]['rand_ch1_ave'] = np.array(ch1_ave)
            data_store[filename]['rand_ch2_ave'] = np.array(ch2_ave)
            data_store[filename]['rand_ch3_ave'] = np.array(ch3_ave)

            data_store[filename]['rand_ch0_sum'] = np.array(ch0_sum)
            data_store[filename]['rand_ch1_sum'] = np.array(ch1_sum)
            data_store[filename]['rand_ch2_sum'] = np.array(ch2_sum)
            data_store[filename]['rand_ch3_sum'] = np.array(ch3_sum)

            data_store[filename]['rand_ch0_ave_norm'] = np.array(ch0_ave_norm)
            data_store[filename]['rand_ch1_ave_norm'] = np.array(ch1_ave_norm)
            data_store[filename]['rand_ch2_ave_norm'] = np.array(ch2_ave_norm)
            data_store[filename]['rand_ch3_ave_norm'] = np.array(ch3_ave_norm)

            data_store[filename]['rand_ch13_pea'] = np.array(ch13_pea)
            data_store[filename]['rand_ch13_pea_flip'] = np.array(ch13_pea_flip_arr)
    return data_store

    
def return_valid_and_random_pts(xpts, ypts, diameter, height, width):

    rn_xpts = []
    rn_ypts = []
    o_xpts = []
    o_ypts = []

    

    
    radius = diameter//2
    for xp,yp in zip(xpts,ypts):
        unq = False
        count = 0
        if yp < height-radius and xp < width-radius and yp > radius and xp > radius:
            o_ypts.append(yp)
            o_xpts.append(xp)
        else:
            continue

        while count < 150 and unq == False:
            unq = True
            nxp = xp + (np.random.random()-0.5)*180
            nyp = yp + (np.random.random()-0.5)*180
            for x0,y0 in zip(xpts,ypts):
                if np.sqrt((nxp-x0)**2 + (nyp-y0)**2) < diameter: #Its the radius of both circles.
                    unq = False
                if nyp > height-radius or nxp > width-radius or nyp < radius or nxp < radius:
                    unq = False

            count += 1
        
            

        rn_xpts.append(np.round(nxp,0).astype(np.int32))
        rn_ypts.append(np.round(nyp,0).astype(np.int32))
    
    return o_xpts, o_ypts, rn_xpts, rn_ypts
def cvrtImg2Dist(im):
    """Converts an intensity image to a distribution."""
    img2 =im
    img2[im < 20] =0
    
    
    #Converts density image into distribution of points.
    colCount =[];
    rowCount =[];
    bin_ratio =2
    img = np.floor(img2/bin_ratio);
    
    for c in range(0,im.shape[0]):
        for r in range(0,im.shape[1]):
            count = int(img[c,r])
            for b in range(0,count):
                colCount.append(c)
                rowCount.append(r)
                
    X1 = np.zeros((colCount.__len__(),2))
    X1[:,1] = colCount
    X1[:,0] = rowCount
    return X1
def meanShift2(X,h,w,minBinSize,bandwidth,init_seeds):
    """Finds peaks using a meanshift clustering algorithm."""
    
    ms = MeanShift(bandwidth=bandwidth,seeds=init_seeds, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    msImg = np.zeros((h,w))

    for k in range(0,n_clusters_):
        my_members = labels == k
        indix = np.where(my_members)
        for c in range(0,indix[0].shape[0]):
            msImg[X[indix[0][c], 1].astype(np.int32), X[indix[0][c], 0].astype(np.int32)] = k+1
    
    return  msImg, labels,cluster_centers
 
def drift_corr(im0,im1,im2,im3,ccx,ccy):
    dis_x = ccx
    dis_y = ccy
    if dis_x<0:
        
        x0 = -dis_x
        x1 = im1.shape[1]-1
        x2 = 0 
        x3 = im1.shape[1]+dis_x -1
    else:
        
        x0 = 0 
        x1 = im1.shape[1]-dis_x-1
        x2 = dis_x
        x3 = im1.shape[1]-1
    if dis_y<0:
        
        y0 = -dis_y
        y1 = im1.shape[0]-1
        y2 = 0 
        y3 = im1.shape[0]+dis_y -1
        
    else:
        
        y0 = 0 
        y1 = im1.shape[0]-dis_y-1
        y2 = dis_y 
        y3 = im1.shape[0]-1
    
    im1_after = im1[int(y0):int(y1),int(x0):int(x1)].astype(np.float64)
    im3_after = im3[int(y2):int(y3),int(x2):int(x3)].astype(np.float64)
    im0_after = im0[int(y2):int(y3),int(x2):int(x3)].astype(np.float64)
    im2_after = im2[int(y2):int(y3),int(x2):int(x3)].astype(np.float64)
    
    return im0_after,im1_after,im2_after, im3_after
def register(im0,im1,im2,im3,ccx,ccy):
    """Function which registers the images: im0 and im2.
    This accounts for drift which occurs between imaging first channel and second channel.
    """
    
    ssize = 20
    shalf = int(ssize/2)
    templ = im0[shalf:-ssize+shalf,shalf:-ssize+shalf]
    templ_mu = np.mean(templ)
    templ_sd = np.std(templ)
    
    out = np.zeros((ssize,ssize))
    pad = ssize
    for r in range(0,ssize):
        for c in range(0,ssize):
            patch = im2[r:-ssize+r,c:-ssize+c]
            patch_mu = np.mean(patch)
            patch_sd = np.std(patch)
            
            out[r,c] = np.sum(((patch-patch_mu)*(templ-templ_mu))/(patch_sd*templ_sd))
    y, x = np.unravel_index(np.argmax(out), out.shape)
    
    #corr = signal.correlate2d(im0, im2, boundary='symm', mode='same')
    
    dis_x = float(x)-shalf+ccx
    dis_y = float(y)-shalf+ccy
    print 'drift correction: pixel shift in x:',dis_x
    print 'drift correction: pixel shift in y:',dis_x
    
    
    
    if dis_x<0:
        
        x0 = -dis_x
        x1 = im0.shape[1]-1
        x2 = 0 
        x3 = im0.shape[1]+dis_x -1
    else:
        
        x0 = 0 
        x1 = im0.shape[0]-dis_x-1
        x2 = dis_x
        x3 = im0.shape[0]-1
    if dis_y<0:
        
        y0 = -dis_y
        y1 = im0.shape[1]-1
        y2 = 0 
        y3 = im0.shape[1]+dis_y -1
        
    else:
        
        y0 = 0 
        y1 = im0.shape[1]-dis_y-1
        y2 = dis_y 
        y3 = im0.shape[1]-1
        
    
    im0_after = im0[int(y0):int(y1),int(x0):int(x1)].astype(np.float64)
    im1_after = im1[int(y0):int(y1),int(x0):int(x1)].astype(np.float64)
    im2_after = im2[int(y2):int(y3),int(x2):int(x3)].astype(np.float64)
    im3_after = im3[int(y2):int(y3),int(x2):int(x3)].astype(np.float64)
    
    return im0_after, im1_after, im2_after, im3_after
def make_figure(im0_after,im1_after,im2_after,im3_after):
    
    
    #Subtract background from the image to remove noise.
    
    im1_afterf1 = filters.gaussian_filter(im1_after ,sigma=60)
    im1_afterf1 = filters.gaussian_filter(im1_after ,sigma=2)-im1_afterf1
    im1_afterf1[im1_afterf1<0] = 0
    
    
    im3_afterf3 = filters.gaussian_filter(im3_after ,sigma=60)
    im3_afterf3 = filters.gaussian_filter(im3_after ,sigma=2)-im3_afterf3
    im3_afterf3[im3_afterf3<0] = 0
    
    
    #im1_afterf1 = im1_after
    #im1_afterf3 = im3_after
    
    
    #Sort the lists and then use them to find the % saturation.
    im1max = np.sort(im1_afterf1.reshape(-1))
    im3max = np.sort(im3_afterf3.reshape(-1))
    im0max = np.sort(im0_after.reshape(-1))
    im2max = np.sort(im2_after.reshape(-1))
    
    #GFP images. 
    im0T = im0_after/ im0max[(int(np.floor(im0max.shape[0]*0.9995)))]
    im2T = im2_after/ im2max[(int(np.floor(im2max.shape[0]*0.9995)))]
    #imRGB[:,:,1] = im2_after/ im2max[(int(np.floor(im2max.shape[0]*0.90)))]
    
    #The pex images.
    imRGB2 = np.zeros((im0_after.shape[0],im0_after.shape[1],3)).astype(np.float64)
    imRGB1 = np.zeros((im0_after.shape[0],im0_after.shape[1],3)).astype(np.float64)
    
    im1T = im1_afterf1/ im1max[int(np.floor(im1max.shape[0]*0.9995))]
    im3T = im3_afterf3/ im3max[int(np.floor(im3max.shape[0]*0.9995))]
    
    im0T[im0T>1] = 1
    im1T[im1T>1] = 1
    im2T[im2T>1] = 1
    im3T[im3T>1] = 1
    #im0T[im0T<0.5] = 0
    #im1T[im1T<0.5] = 0
    #im2T[im2T<0.5] = 0
    imRGB2[:,:,0] = im1T
    imRGB2[:,:,1] = im3T
    imRGB2[:,:,2] = im0T
    
    imRGB1[:,:,0] = im0T
    imRGB1[:,:,1] = im2T
    
    return  imRGB2,imRGB1


