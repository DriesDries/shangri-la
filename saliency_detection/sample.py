'''
    作成した関数たち
'''


    def circles(self,img):
        '''
        画像から円を検出し，円を描画した画像を返す
        '''
        gimg = img
        # a, gimg = cv2.threshold(gimg, 250, 255,THRESH_BINARY_INV)
        cv2.imshow('thresh',gimg)
        circles_img = np.zeros_like(gimg)
        circles = cv2.HoughCircles(gimg, method = cv.CV_HOUGH_GRADIENT, dp = 2, minDist = 10, minRadius = 50, maxRadius = 100)
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(circles_img,center = (i[0],i[1]), radius = i[2], color = (255,255,255),thickness = -1)  # draw the outer circle
            # cv2.circle(circles_img,(i[0],i[1]),2,(0,0,255),3)     # draw the center 

        return circles_img

    def calculate_differ(self,img1,img2):
        '''
        二つの画像の差分を計算して返す
        '''
        # DoGを求める
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        differ = img1 - img2
        for u in range(512):
            for v in range(512):
                if differ[u,v] < 0:
                    differ[u,v] = 0
        # データ型
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)                    
        differ = differ.astype(np.uint8)

        return differ

    def DoG(self,features):
        '''
            入力された各特徴画像に対してDoGを計算する
            args : features -> uint8のMat型1ch画像
            dst  : differs  -> list(各特徴のDoG)
        '''

        differs = range(len(features))
        for i, feature in enumerate(features):
            f1 = cv2.GaussianBlur(feature, ksize = (sm.ksize1,sm.ksize1),sigmaX = sm.sigmaX1)
            f2 = cv2.GaussianBlur(feature, ksize = (sm.ksize2,sm.ksize2),sigmaX = sm.sigmaX2)
        # calculate DoG
            differ = sm.calculate_differ(f1, f2)
            # feature_gau[2*i] = f1
            # feature_gau[2*i+1] = f2
            differs[i] = differ        

        return differs



        
    x = np.arange(0, 512, 1)
    y = np.arange(0, 512, 1)
    X, Y = np.meshgrid(x, y)
    # Z = np.sin(X)+ np.cos(Y)
    # z = np.array(Z)
    Z = img
    # print z.shape

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(X,Y,Z)
    plt.show()

   def DoG(self,pyramids):
        '''
            入力されたpyramidからDoGを求める
            args : pyramids   -> 各階層のgaussian_imgが入ったlist
            dst  : DoG        -> uint8のMat型1ch画像
            param: differs    -> 隣の階層との差分画像が入ってるlist
        '''

        # 空配列の生成
        differs = range(3)
        DoG = 0

        # 差分とるためにpyramidをfloatに
        pyramids = np.array(pyramids).astype(np.float)

        # ピラミッドの前後で差分とる
        for i in range(3):
            differs[i] = np.abs(pyramids[i] - pyramids[i+1]) # 絶対値をとる
            DoG = DoG + differs[i]                           # 差の和がDoGになる

        # 0-255で正規化して，画像表示
        differs = np.array(differs).astype(np.uint8)
        differs = cv2.normalize(differs, 0, 255, norm_type = cv2.NORM_MINMAX)
        for i in range(sm.viewsigmaX):
            cv2.imshow('differ%d'%(i+1),np.array(differs[i]<10, np.uint8)*255)

        # differ = np.abs(pyramids[1] - pyramids[2])

        # DoGを正規化して返す
        DoG = cv2.normalize(DoG.astype(np.uint8), 0, 255, norm_type = cv2.NORM_MINMAX)

        return DoG

    def sobel(self,img):
        '''
        sobelfilterによって直線を検出する
        '''
        gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gimg',gimg)
        grad_x = cv2.Sobel(gimg, cv2.CV_8U, 1, 0, ksize = 1)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        cv2.imshow('grad_x',abs_grad_x)
        

        return abs_grad_x



    def sm(self,img):
        '''
            入力された画像のSaliency Mapを求める

        '''
        # 特徴一覧
        gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_r = img[:,:,2]
        r_minus_g = sm.calculate_differ(img_r, img[:,:,1])

        # 特徴入った配列
        features = [gimg,img_r,r_minus_g]
        features_name = ['gray','red','red - green']

        # それぞれの特徴ごとにDoGを求める
        differs = sm.DoG(features)

        sms = np.zeros_like(gimg)
        for i, differ in enumerate(differs):
            print differ[0,:20], differ.astype(np.float)[0,:20]
            # cv2.imshow('DoG%s  '%i + features_name[i], differ.astype(np.float) / np.amax(differ)) # maxが1となる，もともと0だったら0のまま
            cv2.imshow('DoG%s  '%i + features_name[i], differ)
            sms = sms + differ
        return sms

        


    def normalize(self,differs):
        '''
            入力された画像をSM用に正規化し，正規化された画像を返す
            ok 1.マップ上の値が固定の範囲 [0...M] になるように正規化する。
            2.マップ上の最大値 M を取る場所を検出し、それ以外にある局所的な最大値(極大値)を全て抽出してその平均値 m を求める。
            3.全ての値に ( M - m )2 を掛ける。
        '''
        # 各DoGから極大値を求める，極大値の部分を255とした画像を返す
        for i, differ in enumerate(differs):
            M = np.amax(differ)
            m = sm.calc_maximal_ave(differ)
            print M,m
            # print 'M=',M,'m=',m
            # print (M-m)**2
            # if i == 0:
                # break

    def calc_maximal_ave(self,differ):
        '''
            画像上の極大値を取得し，極大値が255になっている画像と，その座標と値も返す

            param: maximal_mask ->　入力画像の極大値の点が255になっている
                   maximal_img  ->  元画像で極値の点以外は0になってる画像
        ''' 
        xmax_img = np.zeros_like(differ)
        ymax_img = np.zeros_like(differ)

        img = differ

        # x方向に関して
        for v, rows in enumerate(img): #行が入る
            maxId = np.array(signal.argrelmax(rows))
            for u in maxId[0,:]:
                xmax_img[u,v] = 100
        # cv2.imshow('xmax_img',xmax_img)

        # y方向に関して
        for u, cols in enumerate(img): #行が入る
            maxId = np.array(signal.argrelmax(cols))
            for v in maxId[0,:]:
                ymax_img[u,v] = 100
        # cv2.imshow('ymax_img',ymax_img)        
        
        # 両方向足す
        maximal_mask = xmax_img + ymax_img
        a,maximal_mask = cv2.threshold(maximal_mask,160,255, cv2.THRESH_BINARY)
        # cv2.imshow('kyoku',np.array(maximal_img))
        #空画像
        zero = np.zeros_like(differ)
        maximal_total = 0
        maximal_img = cv2.add(differ, zero, mask = maximal_mask)
        a = np.array(np.where(np.array(maximal_img) > 0))
        # print a.shape
        for i in range(len(a[1])):
            # print 'maximal_x =',a[0,i],'y =',a[1,i]
            # maximal_xy = a[:,i] # ->　条件を満たす画像上での座標
            maximal = differ[a[0,i],a[1,i]]
            maximal_total += maximal
            maximal_ave = 1.0 * maximal_total / len(a[1])
            # print a[]
            # print a[:,i]
        # どっかでずれてる
        # cv2.imshow('test',maximal_img)
        # print maximal_total, maximal_ave

        return maximal_ave

    def gau_pyramid(self,gimg):
        '''
            入力された各特徴画像に対してgaussian pyramidを計算する
            args : gimg -> uint8のMat型1ch画像
            dst  : gau_pyramid  -> 各階層のgaussian画像が入ったlist
        '''
        gau_pyramid = range(sm.sigmaX)
        # differs = range(len(gimg))
        for i in range(sm.sigmaX):
            gau_pyramid[i] = cv2.GaussianBlur(gimg, ksize = (sm.ksize,sm.ksize),sigmaX = 2**(i), sigmaY = 0)
        return gau_pyramid        

    def DoG2(self, pyramid):
        '''
            入力されたpyramidからDoGを求める
            args : pyramids   -> 各階層のgaussian_imgが入ったlist
            dst  : DoG        -> uint8のMat型1ch画像のlist
            param: differs    -> 隣の階層との差分画像が入ってるlist
        '''

        # 空配列の生成
        differs = range(6)
        # 差分とるためにpyramidをfloatに
        pyramid = np.array(pyramid).astype(np.float)

        # ピラミッドの差分を取る，絶対値で
        differs[0] = np.abs(pyramid[1] - pyramid[4])
        differs[1] = np.abs(pyramid[1] - pyramid[5]) 
        differs[2] = np.abs(pyramid[2] - pyramid[5]) 
        differs[3] = np.abs(pyramid[2] - pyramid[6]) 
        differs[4] = np.abs(pyramid[3] - pyramid[6]) 
        differs[5] = np.abs(pyramid[3] - pyramid[7]) 

        # for i, differ in enumerate(differs):
            # cv2.imshow('differ%s'%i,differ/np.amax(differs[i]))
            # cv2.imshow('center',pyramid[2].astype(np.uint8))
            # cv2.imshow('surround',pyramid[5].astype(np.uint8))


        # 0-255で正規化して，画像表示
        differs = np.array(differs).astype(np.uint8)
        # for i in range(len(differs)):
            # differs[i] = cv2.normalize(differs[i], 0, 255, norm_type = cv2.NORM_MINMAX)
            # cv2.imshow('differ%d'%(i+1),np.array(differs[i]<10, np.uint8)*255)
            # cv2.imshow('differ%d'%i,np.array(differs[i]))

        DoG = differs
        return DoG

    def filter(self,imgs):
        '''
        任意のkernelでfilter処理をする
        args : img     -> 入力画像,もしくは画像のlist
        dst  : SM      -> saliencymap,uint8のMat型1ch画像
        param: kernel  -> saliency_map.kernelを用いる
        

        '''
        
        if len(imgs) < 100: # 画像が複数枚の場合
            response = range(len(imgs))
            for i, img in enumerate(imgs):
                response[i] = cv2.filter2D(img, cv2.CV_8U, np.array(sm.kernel))
                # cv2.imshow('response%s'%(i+1), response[i])

        else: # 画像が1枚の場合
            response = cv2.filter2D(imgs, cv2.CV_8U, np.array(sm.sobel_x))
            # cv2.imshow('response', response)

        return response        



        
        # rgbMax = cv2.max(b,cv2.max(r,g))
        # rgbMax[rgbMax == 0] = 0.00001 # 0割を防いでる

        # rgMin = cv2.min(r,g)

        # # calculate rg,by
        # rg = (r-g) / rgbMax
        # by = (b - rgMin) / rgbMax
        # rg[rg < 0] = 0
        # by[by < 0] = 0
        
        # return rg, by
    def AveLocalMax(self,img):
        '''
            入力された画像の極大値の平均を求める
            args : img -> uint,1ch
            dst  : AveLocalMax　-> 極大値の平均
        '''
        '''maxima = np.array(signal.argrelmax(IntensityImg))'''

        stepsize = 16
        width = 512
        height = 512

        # find local maximal
        numlocal = 0
        lmaxmean = 0
        for y in range(0, height - stepsize, stepsize):
            for x in range(0, width - stepsize, stepsize):
                localimg = img[y:y+stepsize,x:x+stepsize]
                minval, maxval, minloc, maxloc = cv2.minMaxLoc(localimg)
                lmaxmean += maxval
                numlocal += 1
        AveLocalMax = lmaxmean / numlocal

        return AveLocalMax

        # self.GaborKernel_0 = np.array([\
        # [ 1.85212E-06, 1.28181E-05, -0.000350433, -0.000136537, 0.002010422, -0.000136537, -0.000350433, 1.28181E-05, 1.85212E-06 ],\
        # [ 2.80209E-05, 0.000193926, -0.005301717, -0.002065674, 0.030415784, -0.002065674, -0.005301717, 0.000193926, 2.80209E-05 ],\
        # [ 0.000195076, 0.001350077, -0.036909595, -0.014380852, 0.211749204, -0.014380852, -0.036909595, 0.001350077, 0.000195076 ],\
        # [ 0.000624940, 0.004325061, -0.118242318, -0.046070008, 0.678352526, -0.046070008, -0.118242318, 0.004325061, 0.000624940 ],\
        # [ 0.000921261, 0.006375831, -0.174308068, -0.067914552, 1.000000000, -0.067914552, -0.174308068, 0.006375831, 0.000921261 ],\
        # [ 0.000624940, 0.004325061, -0.118242318, -0.046070008, 0.678352526, -0.046070008, -0.118242318, 0.004325061, 0.000624940 ],\
        # [ 0.000195076, 0.001350077, -0.036909595, -0.014380852, 0.211749204, -0.014380852, -0.036909595, 0.001350077, 0.000195076 ],\
        # [ 2.80209E-05, 0.000193926, -0.005301717, -0.002065674, 0.030415784, -0.002065674, -0.005301717, 0.000193926, 2.80209E-05 ],\
        # [ 1.85212E-06, 1.28181E-05, -0.000350433, -0.000136537, 0.002010422, -0.000136537, -0.000350433, 1.28181E-05, 1.85212E-06 ]])   

        # self.GaborKernel_45 = np.array([\
        # [  4.04180E-06,  2.25320E-05, -0.000279806, -0.001028923,  3.79931E-05,  0.000744712,  0.000132863, -9.04408E-06, -1.01551E-06 ],\
        # [  2.25320E-05,  0.000925120,  0.002373205, -0.013561362, -0.022947700,  0.000389916,  0.003516954,  0.000288732, -9.04408E-06 ],\
        # [ -0.000279806,  0.002373205,  0.044837725,  0.052928748, -0.139178011, -0.108372072,  0.000847346,  0.003516954,  0.000132863 ],\
        # [ -0.001028923, -0.013561362,  0.052928748,  0.460162150,  0.249959607, -0.302454279, -0.108372072,  0.000389916,  0.000744712 ],\
        # [  3.79931E-05, -0.022947700, -0.139178011,  0.249959607,  1.000000000,  0.249959607, -0.139178011, -0.022947700,  3.79931E-05 ],\
        # [  0.000744712,  0.003899160, -0.108372072, -0.302454279,  0.249959607,  0.460162150,  0.052928748, -0.013561362, -0.001028923 ],\
        # [  0.000132863,  0.003516954,  0.000847346, -0.108372072, -0.139178011,  0.052928748,  0.044837725,  0.002373205, -0.000279806 ],\
        # [ -9.04408E-06,  0.000288732,  0.003516954,  0.000389916, -0.022947700, -0.013561362,  0.002373205,  0.000925120,  2.25320E-05 ],\
        # [ -1.01551E-06, -9.04408E-06,  0.000132863,  0.000744712,  3.79931E-05, -0.001028923, -0.000279806,  2.25320E-05,  4.04180E-06 ]])     

        # self.GaborKernel_90 = np.array([\
        # [  1.85212E-06,  2.80209E-05,  0.000195076,  0.000624940,  0.000921261,  0.000624940,  0.000195076,  2.80209E-05,  1.85212E-06 ],\
        # [  1.28181E-05,  0.000193926,  0.001350077,  0.004325061,  0.006375831,  0.004325061,  0.001350077,  0.000193926,  1.28181E-05 ],\
        # [ -0.000350433, -0.005301717, -0.036909595, -0.118242318, -0.174308068, -0.118242318, -0.036909595, -0.005301717, -0.000350433 ],\
        # [ -0.000136537, -0.002065674, -0.014380852, -0.046070008, -0.067914552, -0.046070008, -0.014380852, -0.002065674, -0.000136537 ],\
        # [  0.002010422,  0.030415784,  0.211749204,  0.678352526,  1.000000000,  0.678352526,  0.211749204,  0.030415784,  0.002010422 ],\
        # [ -0.000136537, -0.002065674, -0.014380852, -0.046070008, -0.067914552, -0.046070008, -0.014380852, -0.002065674, -0.000136537 ],\
        # [ -0.000350433, -0.005301717, -0.036909595, -0.118242318, -0.174308068, -0.118242318, -0.036909595, -0.005301717, -0.000350433 ],\
        # [  1.28181E-05,  0.000193926,  0.001350077,  0.004325061,  0.006375831,  0.004325061,  0.001350077,  0.000193926,  1.28181E-05 ],\
        # [  1.85212E-06,  2.80209E-05,  0.000195076,  0.000624940,  0.000921261,  0.000624940,  0.000195076,  2.80209E-05,  1.85212E-06 ]])

        # self.GaborKernel_135 = np.array([\
        # [ -1.01551E-06, -9.04408E-06,  0.000132863,  0.000744712,  3.79931E-05, -0.001028923, -0.000279806, 2.2532E-05, 4.0418E-06 ],\
        # [ -9.04408E-06,  0.000288732,  0.003516954,  0.000389916, -0.022947700, -0.013561362, 0.002373205, 0.00092512, 2.2532E-05 ],\
        # [  0.000132863,  0.003516954,  0.000847346, -0.108372072, -0.139178011, 0.052928748, 0.044837725, 0.002373205, -0.000279806 ],\
        # [  0.000744712,  0.000389916, -0.108372072, -0.302454279,  0.249959607, 0.46016215, 0.052928748, -0.013561362, -0.001028923 ],\
        # [  3.79931E-05, -0.022947700, -0.139178011,  0.249959607,  1.000000000, 0.249959607, -0.139178011, -0.0229477, 3.79931E-05 ],\
        # [ -0.001028923, -0.013561362,  0.052928748,  0.460162150,  0.249959607, -0.302454279, -0.108372072, 0.000389916, 0.000744712 ],\
        # [ -0.000279806,  0.002373205,  0.044837725,  0.052928748, -0.139178011, -0.108372072, 0.000847346, 0.003516954, 0.000132863 ],\
        # [  2.25320E-05,  0.000925120,  0.002373205, -0.013561362, -0.022947700, 0.000389916, 0.003516954, 0.000288732, -9.04408E-06 ],\
        # [  4.04180E-06,  2.25320E-05, -0.000279806, -0.001028923,  3.79931E-05 , 0.000744712, 0.000132863, -9.04408E-06, -1.01551E-06 ]])


def ViewKernel(self,kernel):
        '''
            kernel変えて表示したいとき
        '''
        for i in range(1,10,1):
            x = arange(0, 99, 1)
            y = arange(0, 99, 1)
            X, Y = meshgrid(x, y) 
            z =  cv2.getGaborKernel(ksize = (99,99), sigma = i,theta = 0, lambd = 10, gamma = 1)
            Z = np.array(z)
            print Z.shape
            plt.xlabel('pixel')
            plt.ylabel('pixel')
            plt.title('Kernel Size = 100,'+'  Sigma = %s,'%i+'  Theta = %s,'%0+'  Lambda = 1,'+'  Gamma = 1')
            plt.pcolor(X, Y, Z)
            plt.colorbar()

            plt.pause(.01)
            plt.show()

cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]])
cv2.cornerHarris(src, blockSize, ksize, k[, dst[, borderType]])