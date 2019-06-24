import cv2
from pylab import *
import numpy as np
from numpy import dot
from scipy import signal,linalg,optimize
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from PIL import Image
import math as m
import time
from fast_histogram import histogram1d
import scipy.io


ply_header = '''ply
format ascii 1.0
element vertex %(punti_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face 0
property list uint8 int32 vertex_indices
end_header
'''
ply_header_notexture = '''ply
format ascii 1.0
element vertex %(punti_num)d
property float x
property float y
property float z
element face 0
property list uint8 int32 vertex_indices
end_header
'''

def edge(img):
	x,y,z = img.shape
	b,g,r = cv2.split(img)

	#MASK
	mask = np.ones((x,y))
	for i in range(1,x-1):
		for j in range(1,y-1):
			if r[i,j] > 250 or g[i,j] > 250 or b[i,j] > 250:
				mask[i-1,j-1] = 0
				mask[i,j-1] = 0
				mask[i+1,j-1] = 0
				mask[i-1,j] = 0
				mask[i,j] = 0
				mask[i+1,j] = 0
				mask[i-1,j+1] = 0
				mask[i,j+1] = 0
				mask[i+1,j+1] = 0

	# GRAY EDGE
	hk = [[-1,0,1]]
	vk = [[-1],[0],[1]]

	gr = np.sqrt(np.add(np.power(signal.convolve(r, hk,'same'),2),np.power(signal.convolve(r, vk,'same'),2)))
	#meanedgeR = np.mean(np.multiply(gr,mask))
	gg = np.sqrt(np.add(np.power(signal.convolve(g, hk,'same'),2),np.power(signal.convolve(g, vk,'same'),2)))
	#meanedgeG = np.mean(np.multiply(gg,mask))
	gb = np.sqrt(np.add(np.power(signal.convolve(b, hk,'same'),2),np.power(signal.convolve(b, vk,'same'),2)))
	#meanedgeB = np.mean(np.multiply(gb,mask))


	mnorm = 6.0;
	gr = np.float32(gr)
	gg = np.float32(gg)
	gb = np.float32(gb)
	gr = np.power(gr,mnorm)
	gg = np.power(gg,mnorm)
	gb = np.power(gb,mnorm)

	whiter = np.power(np.multiply(gr,mask).sum(),1/mnorm)
	whiteg = np.power(np.multiply(gg,mask).sum(),1/mnorm)
	whiteb = np.power(np.multiply(gb,mask).sum(),1/mnorm)

	mean = np.sqrt(np.divide(np.add(np.add(np.power(whiter,2),np.power(whiteg,2)),np.power(whiteb,2)),3))
	#print mean

	finalr = np.divide(r, whiter/mean)
	finalg = np.divide(g, whiteg/mean)
	finalb = np.divide(b, whiteb/mean)

	out = np.dstack((finalb,finalg,finalr))

	return out

def calibrareCamere(path_left,path_right,numero_im):

    print 'calibrareCamere'

    findCorner_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                            30, 0.1)

    objp_pattern = np.zeros((6*7,3), np.float32)
    objp_pattern[:,:2] = np.mgrid[0:162:27,0:189:27].T.reshape(-1,2) # pattern con grandezza lato square

    obj_points = []
    img_left_points = []
    img_right_points = []

    for i in range(1,numero_im+1):

        gray_left = cv2.imread(path_left+'Left'+str(i)+'.jpg',0)
        #print type(gray_left)
        gray_right = cv2.imread(path_right+'Right'+str(i)+'.jpg',0)
        #print type(gray_right)
        image_size = gray_left.shape[::-1]

        find_chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH |\
                                cv2.CALIB_CB_NORMALIZE_IMAGE |\
                                cv2.CALIB_CB_FAST_CHECK

        left_found, left_corners = cv2.findChessboardCorners(gray_left, (6,7),
                                                                None)
        right_found, right_corners = cv2.findChessboardCorners(gray_right, (6,7),
                                                                None)

        if left_found and right_found:
            cv2.cornerSubPix(gray_left, left_corners, (11,11), (-1,-1),
                                findCorner_criteria)
            cv2.cornerSubPix(gray_right, right_corners, (11,11), (-1,-1),
                                findCorner_criteria)
            img_left_points.append(left_corners)
            img_right_points.append(right_corners)
            obj_points.append(objp_pattern)

            '''
            cv2.drawChessboardCorners(gray_left, (7,6), left_corners,
                                        left_found)
            cv2.drawChessboardCorners(gray_right, (7,6), right_corners,
                                        right_found)
            cv2.imshow("left chess", gray_left)
            cv2.imshow("right chess", gray_right)
            cv2.waitKey(0)
            '''

    ret, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_points,
                                                            img_left_points,
                                                            image_size,
                                                             None, None)
    ret, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_points,
                                                            img_right_points,
                                                            image_size,
                                                             None, None)

    tot_error_R = 0
    for i in xrange(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecsR[i],
                                            tvecsR[i], mtxR, distR)
        error = cv2.norm(img_right_points[i],imgpoints2,
                            cv2.NORM_L2)/len(imgpoints2)
        tot_error_R += error

    tot_error_L = 0
    for i in xrange(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecsL[i],
                                            tvecsL[i], mtxL, distL)
        error = cv2.norm(img_left_points[i],imgpoints2,
                                            cv2.NORM_L2)/len(imgpoints2)
        tot_error_L += error

    #print "mean error left: ", tot_error_L/len(obj_points)
    #print "mean error right: ", tot_error_R/len(obj_points)

    return (mtxL,distL,mtxR,distR,obj_points,img_left_points,img_right_points)

def undistortImage(imgLeft,imgright,mtxL,distL,mtxR,distR):

    hL,  wL = imgLeft.shape[:2]
    hR, wR = imgright.shape[:2]

    new_camera_matrix_Left, roiLeft = cv2.getOptimalNewCameraMatrix(mtxL, distL,
                                                                        (wL,hL),
                                                                        0, None)
    new_camera_matrix_right, roiright = cv2.getOptimalNewCameraMatrix(mtxR,distR,
                                                                        (wR,hR),
                                                                        0, None)


    #stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    #stereocalib_flags = (cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH
                        #| cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_USE_INTRINSIC_GUESS)

    LeftDist = cv2.undistort(imgLeft, mtxL, distL, None, new_camera_matrix_Left)
    rightDist = cv2.undistort(imgright, mtxR, distR, None, new_camera_matrix_right)

    mtxL = new_camera_matrix_Left
    mtxR = new_camera_matrix_right

    xL,yL,wL,hL = roiLeft
    xR,yR,wR,hR = roiright

    Left = False
    Right = False

    if(roiLeft!=(0,0,0,0)):
        LeftDist = LeftDist[yL:yL+hL, xL:xL+wL]
        #cv2.imshow("LeftDist", LeftDist)
        #cv2.waitKey(0)
        Left = True
    if(roiright!=(0,0,0,0)):
        rightDist = rightDist[yR:yR+hR, xR:xR+wR]
        #cv2.imshow("rightDist", rightDist)
        #cv2.waitKey(0)
        Right = True
    if(Left==True and Right==True):
        print "DISTORTE ENTRAMBI"
        return (LeftDist,rightDist)
    if(Left==True and Right==False):
        print "DISTORTA SINISTRA"
        return (LeftDist,imgright)
    if(Left==False and Right==True):
        print "DISTORTA DESTRA"
        return (imgLeft,rightDist)
    if(Left==False and Right==False):
        print "DISTORTA NESSUNA"
        return (imgLeft,imgright)

def rettificaImmagini(imgLeft,imgright,mtxL,distL,mtxR,distR,image_size,R,T,path):

    R1 = np.zeros((3*3,1))
    R2 = np.zeros((3*3,1))
    P1 = np.zeros((3*4,1))
    P2 = np.zeros((3*4,1))
    Q = None

    R1,R2,P1,P2,Q,roi1,roi2 = cv2.stereoRectify(mtxL, distL, mtxR, distR, image_size,
                                                R, T,
                                                flags =  cv2.CALIB_ZERO_DISPARITY,
                                                alpha =-1)

    # Q = matrice che serve nel metodo reprojectImageTo3D, per trovare disparity #
    # R1,R2 = matrici di rotazione per la prima e seconda camera, per rettificare immagini #
    # P1,P2 = matrici di proiezione nel nuovo sistema per la prima e seconda camera #

    map1x, map1y = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, image_size,
                                                cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, image_size,
                                                cv2.CV_32FC1)

    imgLeftRemap = cv2.remap(imgLeft, map1x, map1y, cv2.INTER_LANCZOS4)
    im = Image.fromarray(imgLeftRemap)
    im.save("output/"+str(path)+"/SinistraRettifica.png")
    imgrightRemap = cv2.remap(imgright, map2x, map2y, cv2.INTER_LANCZOS4)
    im = Image.fromarray(imgrightRemap)
    im.save("output/"+str(path)+"/DestraRettifica.png")

    insiemeRett = np.hstack([imgLeftRemap, imgrightRemap])
    insiemeNoRett = np.hstack([imgLeft, imgright])
    for line in range(0, int(insiemeRett.shape[0] / 20)):
        insiemeRett[line * 20, :] = (0)
    for line in range(0, int(insiemeNoRett.shape[0] / 20)):
        insiemeNoRett[line * 20, :] = (0)

    #plt.imshow(insiemeNoRett, 'gray')
    #plt.show()

    #plt.imshow(insiemeRett,'gray')
    #plt.show()


    return (imgLeftRemap,imgrightRemap,Q)

def stereoMatch(imgLeft,imgRight,path):

    #stereo Correspondence
    window_size = 15
    minDisp = 0
    numDisp = 144
    blockSize = 5
    P1 = 8*2*window_size*2
    P2 = 32*2*window_size**2
    disp12MaxDiff = 1
    preFilterCap = 63
    uniquenessRatio = 7
    speckleWindowSize = 0
    speckleRange = 8

    stereo = cv2.StereoSGBM_create(minDisparity=minDisp,
                                    numDisparities=numDisp,
                                    blockSize=blockSize,
                                    mode=False,
                                    P1=P1,
                                    P2=P2,
                                    uniquenessRatio=uniquenessRatio,
                                    disp12MaxDiff=disp12MaxDiff,
                                    preFilterCap=preFilterCap,
                                    speckleWindowSize=speckleWindowSize,
                                    speckleRange=speckleRange)

    disparity = stereo.compute(imgLeft,imgRight)
    #disparity = (disparity - minDisp) / 16
    disparity = cv2.normalize(disparity, dst=None, alpha=0, beta=255,
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disparity = cv2.medianBlur(disparity,5)
    #plt.imshow(disparity,'gray')
    #plt.show()

    #filtro di smoth
    disparity = cv2.medianBlur(disparity,5)
    im = Image.fromarray(disparity)
    im.convert('RGB').save("output/"+str(path)+"/disparityNuova.png")
    #np.savetxt('output/disaprity.txt',disparity)

    return(disparity)

def pointCloud(disp, im, imgright, mtxL, distL, mtxR, distR, image_size, R, T, Q,path):

    rows,cols,ch = im.shape
    im = rettificaRGB(im, imgright, mtxL, distL, mtxR, distR, image_size, R, T, path)
    texture = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    ### FIND BOX ###
    #print 'findBox'
    mask = findBox(im)

    ### SEGMENTO DEPTH MAP CON MASK BOX ###
    disp_seg = disp*mask
    h, w = image_size
    f = 0.8*w                   # guess for focal length
    Q1 = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])

    ### PROIETTO IN 3D SOLO PUNTI SCATOLA ####
    punti = cv2.reprojectImageTo3D(disp_seg, Q1)

    mask = disp_seg >  0
    punti = punti[mask].reshape(-1,3)
    texture = texture[mask].reshape(-1,3)
    verts = np.hstack([punti, texture])

    # modello reale ma imperfetto
    with open('output/'+str(path)+'/ScatolaOutliers.ply', 'wb') as f:
        f.write((ply_header % dict(punti_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d')

    #print 'elimino outliers'
    punti, texture = deleteOutliers(punti, texture, path)

    return (punti,texture,angle)

def rettificaRGB(im, imgright, mtxL, distL, mtxR, distR, image_size, R, T, path):

    b,g,r = cv2.split(im)

    b, imgRightRemap, Q = rettificaImmagini(b,imgright,mtxL, distL, mtxR,
                                                distR, image_size, R, T, path)
    g, imgRightRemap, Q = rettificaImmagini(g,imgright,mtxL, distL, mtxR,
                                                distR, image_size, R, T, path)
    r, imgRightRemap, Q = rettificaImmagini(r,imgright,mtxL, distL, mtxR,
                                                distR, image_size, R, T, path)

    im = cv2.merge([b, g, r])

    return im

def findBox(img):

    #edge
    copia = img.copy()
    canny = cv2.Canny(img,120,170)

    #taglio parti dove so, a prescindere, che non e presente la scatola
    canny[0:480,0:150] =0
    canny[0:480,510:640] = 0
    #cv2.imshow('canny',canny)
    #cv2.waitKey(0)

    #morfologia, per collegare linee scatola
    kernel = [[1,0,0,1,0,0,1],
              [0,1,0,1,0,1,0],
              [0,0,1,1,1,0,0],
              [1,1,1,1,1,1,1],
              [0,0,1,1,1,0,0],
              [0,1,0,1,0,1,0],
              [1,0,0,1,0,0,1]]
    kernel = np.array((kernel),dtype=np.uint8)
    canny1 = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('linee collegate',canny1)
    #cv2.waitKey(0)

    #contorni
    boh, contours, hierarchy = cv2.findContours(canny1,cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_NONE) # tengo tutti i punti del contorno

    #riconosco contorno scatola
    if len(contours) > 0:
        #trovo il contorno piu largo con minEnclosingCircle
        c = sorted(contours, key = cv2.contourArea, reverse = True)[0] # ordino i contorni per area e prendo il piu grosso
        row,column,depth = c.shape

    #disegno solo contorno scatola, piu o meno
        cv2.drawContours(copia, c, -1, (255,255,255), 1)
        #cv2.imshow('solo scatola',copia)
        #cv2.waitKey(0)

    #riempio interno maschera
    ret,mask = cv2.threshold(copia[:,:,0],244,255,cv2.THRESH_BINARY)
    im_floodfill = np.uint8(mask.copy())
    h, w = mask.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    mask = cv2.bitwise_not(im_floodfill)
    #cv2.imshow("Floodfilled Image", mask)
    #cv2.waitKey(0)

    #morfologia per eliminare cose di troppo da contorno scatola
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)/255

    #immagine segmentata
    b,g,r = cv2.split(img)
    b = b*mask
    g = g*mask
    r = r*mask
    img_seg = cv2.merge((b,g,r))
    #cv2.imshow("immagine segmentata", img_seg)
    #cv2.waitKey(0)

    return mask

def deleteOutliers(punti,texture,path):

    num_punti_prec = punti.shape[0]

    X = punti[:,0]
    Y = punti[:,1]
    Z = punti[:,2]

    X_nuovi = np.asarray([])
    Y_nuovi = np.asarray([])
    Z_nuovi = np.asarray([])

    R = texture[:,0]
    G = texture[:,1]
    B = texture[:,2]

    R_nuovi = np.asarray([])
    G_nuovi = np.asarray([])
    B_nuovi = np.asarray([])

    mean = np.mean(Z, axis=0)
    sd = np.std(Z, axis=0)

    for i in range(0,Z.shape[0]):

        z = Z[i]

        if ((z > mean - 1.75 * sd) & (z < mean + 1.75 * sd)):
            X_nuovi = np.hstack([X_nuovi, X[i]])
            Y_nuovi = np.hstack([Y_nuovi, Y[i]])
            Z_nuovi = np.hstack([Z_nuovi, Z[i]])
            R_nuovi = np.hstack([R_nuovi, R[i]])
            G_nuovi = np.hstack([G_nuovi, G[i]])
            B_nuovi = np.hstack([B_nuovi, B[i]])

    punti = cv2.merge([X_nuovi, Y_nuovi, Z_nuovi])
    texture = cv2.merge([R_nuovi ,G_nuovi , B_nuovi])

    num_punti_nuovi = punti.shape[0]

    verts = np.hstack([punti, texture]).reshape(-1,6)

    #if(num_punti_prec - num_punti_nuovi > 0):
        #print 'Ho eliminato ' + str(num_punti_prec - num_punti_nuovi) + ' outliers'
    #else:
        #print 'Non ho eliminato outliers'

    # modello reale ma imperfetto
    with open('output/'+str(path)+'/ScatolaNoOutliers.ply', 'wb') as f:
        f.write((ply_header % dict(punti_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d')

    return (punti,texture)

def save(mtxL,distL,mtxR,distR,R,T):

    np.savetxt('matrici/mtxL.txt', mtxL)
    np.savetxt('matrici/distL.txt', distL)
    np.savetxt('matrici/mtxR.txt', mtxR)
    np.savetxt('matrici/disRL.txt', distR)
    np.savetxt('matrici/R.txt', R)
    np.savetxt('matrici/T.txt', T)

def read():

    mtxL = np.loadtxt('matrici/mtxL.txt')
    distL = np.loadtxt('matrici/distL.txt')
    mtxR = np.loadtxt('matrici/mtxR.txt')
    distR = np.loadtxt('matrici/disRL.txt')
    R = np.loadtxt('matrici/R.txt')
    T = np.loadtxt('matrici/T.txt')

    return (mtxL,distL,mtxR,distR,R,T)

def findParametri(punti,metodo):

	tolStep = 0.05

	if metodo=='min':
		x0 = np.asarray([[1],[0.5],[0.5]])
		res = optimize.fmin(raddrizzamento,x0,args=(punti,tolStep), xtol=1e-8, ftol=1e-8, disp=0, full_output=1)
		t1 = res[0][0]
		t2 = res[0][1]
		t3 = res[0][2]
		fval = res[1]
	if metodo=='bianco':
		t1,t2,t3,fval = chiamataRaddrizzamento(punti)

	return (t1,t2,t3,fval)

def raddrizzamento(x,P,tolStep):
	# input : x vettore 1x3 dei tre angoli
	#         P point cloud originale
	#         draw bool per sapere se mostare grafico
	#         tolStep intervalli in istogramma

	n = P.shape[0]
	P = P.reshape(n,3)

	x = np.absolute(x)

	t1 = x[0]
	t2 = x[1]
	t3 = x[2]

	M1 = np.array([[1,0,0,0],
	      [0,np.cos(t1),-np.sin(t1),0],
	      [0,np.sin(t1),np.cos(t1),0],
	      [0,0,0,1]]).astype(float)

	M2 = np.array([[np.cos(t2),0,np.sin(t2),0],
	      [0,1,0,0],
	      [-np.sin(t2),0,np.cos(t2),0],
	      [0,0,0,1]]).astype(float)

	M3 = np.array([[np.cos(t3),-np.sin(t3),0,0],
	      [np.sin(t3),np.cos(t3),0,0],
	      [0,0,1,0],
              [0,0,0,1]]).astype(float)

	P = np.hstack([P,np.ones((n,1),dtype=int)]).astype(float)

	P = np.dot(np.dot(np.dot(P,M1),M2),M3)
	temp =  P[:,3].reshape(n,1)
	P[:,0:3] = P[:,0:3] / linalg.kron(np.ones((1,3)),temp) #repmat matlab

	binHist = np.linspace(-4,4,num=160)

	h1, bin_edges1, _ = plt.hist(P[:,0],bins=binHist)
	h2, bin_edges2, _ = plt.hist(P[:,1],bins=binHist)
	h3, bin_edges3, _ = plt.hist(P[:,2],bins=binHist)
	h1 = h1 / float(np.sum(h1))
	h2 = h2 / float(np.sum(h2))
	h3 = h3 / float(np.sum(h3))
	f = np.max(h1)+np.max(h2)+np.max(h3)
	f = -f

	return (t1,t2,t3,f) # importante perche minimize ritorna uno scalare, ritorna (t1,t2,t3,f) con chiamataRaddrizzamento

def rotateTraslate(res,P,T,fileName,path):

	n = P.shape[0]

	t1 = res[0]
	t2 = res[1]
	t3 = res[2]

	M1 = np.array([[1,0,0,0],
	      [0,np.cos(t1),-np.sin(t1),0],
	      [0,np.sin(t1),np.cos(t1),0],
	      [0,0,0,1]])

	M2 = np.array([[np.cos(t2),0,np.sin(t2),0],
	      [0,1,0,0],
	      [-np.sin(t2),0,np.cos(t2),0],
	      [0,0,0,1]])

	M3 = np.array([[np.cos(t3),-np.sin(t3),0,0],
	      [np.sin(t3),np.cos(t3),0,0],
	      [0,0,1,0],
	      [0,0,0,1]])

	P = P.reshape(n,3)
	P = np.hstack([P,np.ones((n,1),dtype=int)])

	P = myDot1(myDot1(myDot1(P,M1),M2),M3)

	temp =  P[:,3].reshape(n,1)
	P[:,0:3] = P[:,0:3] / linalg.kron(np.ones((1,3)),temp)

	#sposto origine in centroide
	centroide = np.mean(P[:,0:3],axis=0)# centroide 1x3
	P[:,0:3] = P[:,0:3] - np.tile(centroide, (n, 1)) #tile tipo repmat matlab, ripeto array tot volte

	#sposto point cloud
	min_x = P[:,0].min()
	min_y = P[:,1].min()
	min_z = P[:,2].min()
	distance_x = abs(0 - min_x)
	distance_y = abs(0 - min_y)
	distance_z = abs(0 - min_z)
	P[:,0] = P[:,0] + distance_x
	P[:,1] = P[:,1] + distance_y
	P[:,2] = P[:,2] + distance_z

	#abbasso in modo che i punti partano da y=0(quota)
	min_y = P[:,1].min()
	P[:,1] = P[:,1] - min_y

	T = T.reshape(n,3)
	verts = np.hstack([P[:,0:3],T])

	with open('output/'+str(path)+'fmin'+fileName, 'wb') as f:
		f.write((ply_header % dict(punti_num=len(verts))).encode('utf-8'))
		np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

	return P[:,0:3]

def findVolume(punti):

	x = punti[:,0]
	y = punti[:,1]
	z = punti[:,2]

	x = np.around(x,decimals=2)
	y = np.around(y,decimals=2)
	z = np.around(z,decimals=2)

	altezza = np.around(al(punti) * 21.4,decimals=2)
	larghezza = np.around(lar(punti)* 21.4,decimals=2)
	profondita = np.around(prof(punti)* 21.4,decimals=2)
	volume = altezza * larghezza * profondita

	return volume

def al(x,y,z):

	altezza = 0
	totPunti = 0

	mx = np.max(y)
	Y = np.sort(y)[::-1]

	while(Y.size):

		Y_temp = Y[np.where(Y==mx)]
		n_temp = Y_temp.shape[0]

		if(n_temp>totPunti):
			totPunti = n_temp
			altezza = np.around(mx, decimals = 2)

		Y = np.delete(Y,range(n_temp),0)
		if(Y.size):
			mx = np.max(Y)

	return altezza

def lar(x,y,z):

	totPunti = 0
	larghezza = 0

	mn = np.min(z)
	index = z.argsort()
	Z = np.sort(z)
	X = x[index]

	while(Z.size):

		Z_temp = Z[np.where(Z==mn)]
		n_temp = Z_temp.shape[0]
		X_temp = X[0:n_temp]

		larghezzaTemp = np.around(np.max(X_temp) - np.min(X_temp), decimals = 2)

		if(n_temp>totPunti):
			totPunti = n_temp
			larghezza = larghezzaTemp

		X = np.delete(X,range(n_temp),0)
		Z = np.delete(Z,range(n_temp),0)
		if(Z.size):
			mn = np.min(Z)

	return larghezza
def prof(x,y,z):

	Xtemp = 0
	totPunti = 0
	profondita = 0

	m = np.min(x)
	index = x.argsort()
	X = np.sort(x)

	while(X.size):

		X_temp = X[np.where(X==m)]
		n_temp = X_temp.shape[0]

		if(n_temp>totPunti):
			totPunti = n_temp
			Xtemp = m

		X = np.delete(X,range(n_temp),0)
		if(X.size):
			m = np.min(X)

	Z = z[np.where(x==Xtemp)]
	profondita = np.around(np.max(Z) - np.min(Z), decimals = 2)

	return profondita

def chiamataRaddrizzamento(punti):

	tolStep = 0.05
	angleStep = 0.1
	F1 = [10, 10, 10, 10]
	r = np.arange(0,1+angleStep,angleStep)
	for t1 in r:
		for t2 in r:
			for t3 in r:
				x = np.reshape(np.asarray([[t1,t2,t3]]),(3,1))
				F1temp = raddrizzamento(x,punti,tolStep)
				F1 = np.vstack([F1,F1temp])

	u = nonzero(F1[:,3]==np.min(F1[:,3]))[0][0]
	#t1,t2,t3,fval = raddrizzamento(F1[u,0:3],punti,tolStep)

	angleStep = 0.01
	F2 = [10, 10, 10, 10]
	r1 = np.arange(F1[u,0]-0.1,F1[u,0]+0.1+0.01,angleStep)
	r2 = np.arange(F1[u,1]-0.1,F1[u,1]+0.1+0.01,angleStep)
	r3 = np.arange(F1[u,2]-0.1,F1[u,2]+0.1+0.01,angleStep)
	for t1 in r1:
		for t2 in r2:
			for t3 in r3:
				x = np.reshape(np.asarray([[t1,t2,t3]]),(3,1))
				F2temp = raddrizzamento(x,punti,tolStep)
				F2 = np.vstack([F2,F2temp])

	u = nonzero(F2[:,3]==np.min(F2[:,3]))[0][0]
	t1,t2,t3,fval = raddrizzamento(F2[u,0:3],punti,tolStep)

	return (t1,t2,t3,fval)
