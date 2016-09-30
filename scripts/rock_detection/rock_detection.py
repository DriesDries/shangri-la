# -*- coding: utf-8 -*-
'''
    Small Rock Detection based on Viola-Jones and Region Growing Algolithm
    Viola-Jones法に基づいて複数のスケールのカーネルによるテンプレートマッチングを行うことで、岩領域の抽出を行う。
    TMの結果に対して閾値処理を行い、その各領域から最大値とそのときに用いたテンプレートを求める。
    それらのピクセルを領域拡張法の種として、その種を中心としてテンプレートの大きさに基づいたガウス分布を展開し、
    エネルギー関数を導入し、定めた閾値よりも小さい場合は、隣のピクセルと結合する。

    Usage: $ python rock_detection.py
    argv : img       -> 3ch画像
           param     ->
    dst  : ror       -> Region of Rocks Image -> 1ch binary image
           seed_img  -> Seed image for image segmentation

    同一のフィルタでやって、閾値した後の各領域の『大きさ』で見た方がいい？

    PT
        - 近傍どこまで見るか
        - 閾値
    小さい領域と大きい領域で分けるかどうか
    画像の領域の何%が岩か

''' 

import time
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import skimage.measure as sk
from scipy import stats
from sklearn import cluster

from mpl_toolkits.axes_grid1 import make_axes_locatable
import filterbank as fb

def main(img, param):
    
    ## Read class
    cv = ImageConvolution()
    rg = RegionGrowing()
    ta = TextureAnalysis()

    if param == None:
        param = [167, np.pi, 0.12]
        print 'param = None : {}'.format(param)
    
    ## Read parameters
    thresh    = param[0]
    direction = param[1]
    sigma     = 4
    bias      = 0
    
    ## Seed Acquisition by Viola-Jones
    cvmaps = cv.convolution(img, direction, sigma)
    seed_img, seed_list, scale_img = cv.get_seed(img, cvmaps, thresh) # listはy,x
    new_seed_img, new_seed_list = cv.twin_seed(img, seed_img, scale_img) # light->255, shade ->130

    ## Texture analysis
    responses = ta.filtering(img, name='MR', radius=2)
    dis, center = ta.var(img, responses[:])

    # texton_map, centers = ta.texton_map(responses, N = 8)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    # a = ax1.imshow(dis)
    # fig.colorbar(a, cax=cax)
    # plt.show()

    ## Image clustering
    ror = rg.main(img, new_seed_list, scale_img, responses, param, dis)

    return ror, seed_img

class ImageConvolution():

    def convolution(self, img, direction, sigma):
        '''
            入力画像を太陽方向に基づいた複数のカーネルで畳みこむ
            param: psi -> 位相
        '''
        ## Kernels acquisition
        kernels = map(lambda x: cv2.getGaborKernel(ksize = (x,x), sigma = sigma, theta = direction, lambd = x, gamma = 25./x, psi = np.pi * 1/2), range(5, 25, 2))

        ## Normalize each kernels -1 ~ 1
        for i,kernel in enumerate(kernels):
            kernels[i] = 1. * kernels[i] / np.amax(kernels[i])

        ## Normalize each response 0 ~ 255 ## Convolution and normalization with kernel size 
        responses = map(lambda x: cv2.filter2D(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), cv2.CV_64F, x),kernels)
        responses = map(lambda x: responses[x]/(kernels[x].shape[0]**2), range(len(range(5, 25, 2))))
        responses = cv2.normalize(np.array(responses), 0, 255, norm_type = cv2.NORM_MINMAX).astype(np.uint8)

        ### display ###
        # for kernel in kernels:
            # cv2.imshow('kernel',abs(kernel))
            # cv2.waitKey(-1)
        # for res in responses:
            # cv2.imshow('responses',res)
            # cv2.waitKey(-1)

        return responses

    def get_seed(self, img, responses, thresh):
        '''
        responsesから，種を選択する。
        seed_imgとseed_listとそれぞれのscaleを返す。
        
        seed_img    : seedにresponsesの値が格納された画像
        seed_list   : seed_imgをlistにしたもの
        seed_list2  : 種が類似度の昇順で並んでるlistスケールが入ってるリスト
        scale       : seed_listに対応したそれぞれのkernelの
        scale_img   : それぞれのピクセルの最も大きいscaleが入ってる
        '''
        
        ## Thresholding
        responses[responses<thresh]=0
        # for res in responses:
            # cv2.imshow('responses',res)
            # cv2.waitKey(-1)


        maxima = np.zeros_like(responses[0])
        scale_img  = np.zeros_like(responses[0])

        ## 各ピクセルの最大値抽出、その値とスケールを画像に入れる
        for i in range(responses.shape[1]):
            for j in range(responses.shape[2]):
                maxima[i,j] = np.max(responses[:,i,j])   # 最大値が入る
                scale_img[i,j]  = np.argmax(responses[:,i,j])+1 # そのときのスケールが入る

        ## Seed Acquisition
        # seed_img = self.get_maxima(maxima) ## 各領域の最大値の抽出
        seed_img = self.get_maxima_scale(maxima, scale_img) ## 各領域の最大のスケールの抽出
        scale_img[seed_img == 0] = 0


        ## 昇順の要素番号の取得
        
        ## listをorderに沿った昇順にする
        seed_list, value = self.img2list(seed_img)
        seed_list2 = []
        order = np.argsort(value)[::-1]
        for i in range(len(seed_list)):
            seed_list2.append(seed_list[order[i]])

        return seed_img, np.array(seed_list2), scale_img

    def get_maxima(self, src):
        '''
        入力された画像を領域分割し、各領域の最大値を算出する。
        src: 1ch-img
        dst: 領域の最大値のピクセルにのみその値が格納された画像。
        '''

        img = src.copy()
        img[img!=0] = 255

        # 各領域にラベルをつける
        labels, num = sk.label(img, return_num = True) 

        seed_img = np.zeros_like(src)
        
        # 各領域の最大値を求める
        for i in range(1,num+1):

            # iの領域だけ残す
            img = copy.deepcopy(src) # 初期に戻す
            img[labels!=i] = 0 # これで残った領域の最大値求める
            
            # 最大値を求める,1行にしたときの値が出てくるからこんなになってる
            y = np.argmax(img)/len(img)
            x = np.argmax(img)%len(img)

            if img[y,x] != 0: # 中に空いた穴じゃなければ種にする
                seed_img[y,x] = src[y,x]
        
        return seed_img

    def get_maxima_scale(self, maxima, scale):
        '''
        入力された画像を領域分割し、各領域の最大値を算出する。
        src: 1ch-img
        dst: 領域の最大値のピクセルにのみその値が格納された画像。
        '''

        # 各領域にラベルをつける
        s = maxima.copy()
        s[s!=0] = 255
        
        labels, num = sk.label(s, return_num = True) 

        seed_img = np.zeros_like(maxima)
        
        # 各領域の最大値を求める
        for i in range(1,np.max(labels)):
            ## 初期化
            scale_i = scale.copy() # 初期に戻す
            maxima_i = maxima.copy()

            ## iの領域だけ残す
            scale_i[labels!=i] = 0 # これで残った領域の最大値求める
            maxima_i[labels!=i] = 0 # これで残った領域の最大値求める

            max_scale_i = np.max(scale_i) # 領域の中の最も大きいスケールを種とする
            maxima_i[scale_i != max_scale_i] = 0
            # print np.max(maxima_i)

            seed_img[maxima_i == np.max(maxima_i)] = np.max(maxima_i)

        return seed_img

    def img2list(self, img):
        ''' 
            画像で非0の座標をlistに
        '''
        vj_list = []
        # vj_list = np.empty((0,2),np.int16)
        value = []

        for i in range(len(img)):
            for j in range(len(img)):
                if img[j,i] != 0:
                    vj_list.append([j,i])
                    value.append(img[j,i])
        
        return vj_list, np.array(value)

    def twin_seed(self, img, seed_img, scale_img):
        '''
            近傍で最も輝度値が高いピクセルと低いピクセルを種とする
        '''
        gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        new_seed = np.zeros_like(gimg)
        new_seed_list = []

        for j in range(4,seed_img.shape[0]-4):
            for i in range(4,seed_img.shape[1]-4):

                if seed_img[j,i] != 0:
                    ## 近傍領域の最大値と最小値をseedにする
                    ma = np.argmax(gimg[j-2:j+3,i-2:i+3])
                    mi = np.argmin(gimg[j-2:j+3,i-2:i+3])
                    new_seed[j-2 + ma/5, i-2 + ma%5] = 255
                    new_seed[j-2 + mi/5, i-2 + mi%5] = 130
                    new_seed_list.append([j,i,j-2 + ma/5, i-2 + ma%5,j-2 + mi/5, i-2 + mi%5])

        return new_seed, np.array(new_seed_list)

class TextureAnalysis():
    
    def filtering(self, img, name, radius):
        
        ## get filterbank
        bank = fb.main(name, radius)
        
        fgimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float64)
        responses = []
        sums = []
        max_responses = []

        if name == 'MR':

            edges = bank[0]
            bars  = bank[1]
            rots  = bank[2]
            
            # gabor filterから3つ
            for i,kernel in enumerate(edges):

                response = cv2.filter2D(fgimg, cv2.CV_64F, kernel)
                responses.append(cv2.filter2D(fgimg, cv2.CV_64F, kernel))
                sums.append(response.sum())
                    
                if (i+1)%6 == 0 :
                    # 最大値がふくまれているresを保存する？
                    # print np.argmax(sums),np.max(sums)
                    max_responses.append(responses[np.argmax(sums)])

                    # 初期化
                    responses = []
                    sums = []

            for i,kernel in enumerate(bars):
                response = cv2.filter2D(fgimg, cv2.CV_64F, kernel)
                responses.append(cv2.filter2D(fgimg, cv2.CV_64F, kernel))
                sums.append(response.sum())
                    
                if (i+1)%6 == 0 :
                    # 最大値がふくまれているresを保存する？
                    # print np.argmax(sums),np.max(sums)
                    max_responses.append(responses[np.argmax(sums)])

            for i,kernel in enumerate(rots):
                max_responses.append(cv2.filter2D(fgimg, cv2.CV_64F, kernel))


        if name == 'LM':
            schmids  = bank[3]
            print schmids.shape
            for i,kernel in enumerate(schmids):
                max_responses.append(cv2.filter2D(fgimg, cv2.CV_64F, kernel))

        ## normalize 0 ~ 1
        max_responses += abs(np.min(max_responses))
        max_responses = max_responses / np.max(max_responses)

        return max_responses

    def texton_map(self, responses, N = 8):
        '''
            k-meansでresponsesをクラスタリングする
        '''
        
        ## 配列の作成
        arr = []
        for res in responses:
            arr.append(res.flatten())
        arr = np.array(arr).T

        ## サンプリングする場合
        # arr = np.array(random.sample(arr,1000))
        # arr = arr[:,0:8] # test function

        ## arrをクラスタリングする
        kmean = cluster.KMeans(n_clusters=N, init='k-means++', n_init=10, max_iter=300,tol=0.0001,precompute_distances='auto', verbose=0,random_state=None, copy_x=True, n_jobs=1)
        arr_cls = kmean.fit(arr)
        pred_labels  = arr_cls.labels_
        pred_centers = arr_cls.cluster_centers_
        
        # texton mapの生成
        size = int(math.sqrt(len(pred_labels)))
        texton_map = np.reshape(pred_labels, (size,size))

        return texton_map, pred_centers

    def var(self, img, responses):
        
        center = []
        dis = np.zeros_like(img[:,:,0]).astype(np.float64)
        
        ## Center Acquitision    
        for n in range(responses.shape[0]):
            res = responses[n,:,:].sum() / 400**2
            center.append(res)

        ## Var processing
        for j in range(dis.shape[0]):
            for i in range(dis.shape[1]):
                dis[j,i] = abs(np.linalg.norm(responses[:,j,i] - center[:]))

        ## Normalization
        dis = dis / np.max(dis)

        return dis, center

class RegionGrowing():

    def main(self, img, seed_list, scale_img, responses, param, dis):
        '''
            Region Growingのメイン
            args : src          -> img,3ch
                   x,y          -> seed pointの座標
                   scale        -> 画像形式
            dst  : region_map   -> 領域が255,他が0の1ch画像
                   masked_img   -> srcにmaskを重ねた画像
            param:      -> 
        '''        

        gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
        rors = np.zeros_like(gimg).astype(np.uint8)
        gauimgs = self.get_gau_image() # 0-1に正規化,float64 # 複数スケールのgaussian imageの用意

        gimg = 1. * gimg / np.max(gimg) 

        for i, seed in enumerate(seed_list):

            # if rors[seed[0],seed[1]] == 0: # 新たな種
            ror = self.growing(gimg, seed, gauimgs[scale_img[seed[0],seed[1]] - 1], responses, param, dis) 
            ror_shade = self.growing_shade(gimg, seed, gauimgs[scale_img[seed[0],seed[1]] - 1], responses, param, dis) 
            rors[ror==255] = 255
            rors[ror_shade==255] = 255

            # print '{}/{} : region size = {}'.format((i+1), len(seed_list[:,0]), np.count_nonzero(ror)+np.count_nonzero(ror2))

        return rors

    def growing(self, img, seed, gauimg, responses, param, dis):
        '''
            args : src          -> img,3ch
                   ori_seed     -> x,yの順の配列
            dst  : region_map   -> 領域が255,他が0の1ch画像
                   masked_img   -> srcにmaskを重ねた画像

            param:      -> 
        '''

        th = param[2]

        # 準備
        ror = np.zeros_like(img).astype(np.uint8)
        ror[seed[0], seed[1]] = 255
        ror[seed[2], seed[3]] = 255
        ror[seed[4], seed[5]] = 255

        sy = seed[0]
        sx = seed[1]
        seeds = []
        seeds.append([seed[2],seed[3]]) # maxを種に
        # seeds.append([seed[4],seed[5]]) # minを種に

        chi1 = 0
        chi2 = 0
        count1 = 0
        count2 = 0

        # region growing
        for i in xrange(100000):

            if i == len(seeds): break # 終了条件
            if i >= 3000: break # 中断条件

            y, x = seeds[i] # renew seed point

            ## Compare with around pixels
            for u,v in zip([x,x-1,x+1,x],[y-1,y,y,y+1]):
              
                ## 中断条件 : 画像の端か検出済みだったら
                if u < 0 or u >= img.shape[1] or v < 0 or v >= img.shape[0]: continue
                elif ror[v,u] != 0: continue

                else: ## 継続条件
                    ## Calculate gaussian value
                    if abs(v - sy) > int(gauimg.shape[0]/2 -1) or abs(u - sx) > int(gauimg.shape[1]/2 -1):
                        gau = 1.0                
                    else:
                        gau = gauimg[int(v-sy+gauimg.shape[0]/2), int(u-sx+gauimg.shape[1]/2)]

                    ''' 領域拡張条件 ''' ## v,u -> 拡張先 y,x -> 今いるseed seed[2],seed[3] -> もともとのseed
                    E2 = gau * abs(img[v,u] - img[tuple(seeds[0])])
                    E3 = img[v,u] < 60
                    E8 = dis[v-1:v+2,u-1:u+2].sum() / 9.
                    if E2 / E8 < th:

                        ror[v,u] = 255
                        seeds.append([v,u])

        ## dilateとerodeをする
        ror = cv2.dilate(ror, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
        ror = cv2.erode(ror, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

        return ror


    def growing_shade(self, img, seed, gauimg, responses, param, dis):
        '''
            args : src          -> img,3ch
                   ori_seed     -> x,yの順の配列
            dst  : region_map   -> 領域が255,他が0の1ch画像
                   masked_img   -> srcにmaskを重ねた画像

            param:      -> 
        '''

        th = param[2]

        # 準備
        ror = np.zeros_like(img).astype(np.uint8)
        ror[seed[0], seed[1]] = 255
        ror[seed[2], seed[3]] = 255
        ror[seed[4], seed[5]] = 255

        sy = seed[0]
        sx = seed[1]
        seeds = []
        # seeds.append([seed[2],seed[3]]) # maxを種に
        seeds.append([seed[4],seed[5]]) # minを種に

        chi1 = 0
        chi2 = 0
        count1 = 0
        count2 = 0

        '''lightとshadeでそれぞれgrowingする'''

        # region growing
        for i in xrange(100000):

            if i == len(seeds): break # 終了条件
            if i >= 3000: break # 中断条件

            y, x = seeds[i] # renew seed point

            ## Compare with around pixels
            for u,v in zip([x,x-1,x+1,x],[y-1,y,y,y+1]):
              
                ## 中断条件 : 画像の端か検出済みだったら
                if u < 0 or u >= img.shape[1] or v < 0 or v >= img.shape[0]: continue
                elif ror[v,u] != 0: continue



                else: ## 継続条件
                    if abs(v - sy) > int(gauimg.shape[0]/2 -1) or abs(u - sx) > int(gauimg.shape[1]/2 -1):
                        gau = 1.0                
                    else:
                        gau = gauimg[int(v-sy+gauimg.shape[0]/2), int(u-sx+gauimg.shape[1]/2)]
                    
                    ''' 領域拡張条件 ''' ## v,u -> 拡張先 y,x -> 今いるseed seed[2],seed[3] -> もともとのseed
                    E2 = gau * abs(img[v,u] - img[tuple(seeds[0])])
                    E3 = img[v,u] < 60
                    E8 = dis[v-1:v+2,u-1:u+2].sum() / 9.
                    if E2 / E8 < th:                    # E3 = 
                        ror[v,u] = 255
                        seeds.append([v,u])

        ## dilateとerodeをする
        ror = cv2.dilate(ror, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
        ror = cv2.erode(ror, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

        return ror

    def get_gau_image(self):
        '''
            scaleの配列に基づいた複数のガウシアン関数を用意する
        '''

        kernels = []
        # sigma = range(12,2,-1) # 分散の定義

        for i in range(10):
            size = 40 # kernelのサイズ
            # nsig = sigma[i]  # 分散sigma
            nsig = 8  # 分散sigma
            interval = (2*nsig+1.)/(size)
            x = np.linspace(-nsig-interval/2., nsig+interval/2., size+1) # メッシュの定義

            kern1d = np.diff(st.norm.cdf(x)) # 多分ここでガウシアンにしてる,1次元
            kernel_raw = np.sqrt(np.outer(kern1d, kern1d)) # 二次元にしてる
            kernel_raw = cv2.normalize(kernel_raw, 0, 1, norm_type = cv2.NORM_MINMAX)
            kernel_raw = abs(1-kernel_raw)
            kernels.append(kernel_raw)

        return kernels

if __name__ == '__main__':


    # filename = 'spirit006-1.png'
    filename = 'spirit118-1.png'

    img = cv2.imread('../../../data/g-t_data/resized/{}'.format(filename))
    # img = cv2.imread('../../../data/test_image/sample/image3.png')
    # img = cv2.resize(img,(500,500))
    true_ror = cv2.imread('../../../data/g-t_data/label/{}'.format(filename),0)
    print 'Target image : {}'.format(filename)

    ## Main processing
    ror, seed_img = main(img, param=None)

    ## Draw result 
    b,g,r = cv2.split((img*0.8).astype(np.uint8))
    # b,g,r = cv2.split(img)
    r[ror == 255] = 255
    # g[true_ror == 255] = 200
    # b[seed_img != 0] = 255
    res = cv2.merge((b,g,r))

    cv2.imshow('res',res)
    plt.pause(1)
    cv2.waitKey(-1)



    # chi = stats.chisquare(responses[:,v,u], responses[:,seed[2],seed[3]])[0]
    # ed = np.linalg.norm(responses[:,v,u] - responses[:, y, x], ord=None) # ordは正規化
    # E1 = abs(img[v,u] - img[tuple(seeds[0])])
    # E4 = abs(stats.chisquare(responses[:,v,u], responses[:,y,x])[0])
    # E5 = dis[texton_map[v,u], texton_map[y,x]] # cluster間の距離
    # E6 = img[v,u] < img[y,x]
    # E7 = dis[v,u]
    
    # if E2 < th and ed > th2:
    # print E2 / dis[v,u]
    # if 1. * E2 / dis[v,u] < th:
    # if E2 < 0.04:
