# -*- coding: utf-8 -*-

'''
使わなくなった関数を入れとく
'''



    # インスタンスメソッド
    # クラスメソッドではクラス変数にアクセスできる
    # じゃあインスタンス変数のある意味は、メモリとかの問題なのか？
    # インスタンス変数とクラス変数の対応関係は、ローカル変数とグローバルと同じなのか？
    # インスタンス変数とクラス変数の対応関係は、メソッド間での継承とクラス間での継承みたいなイメージなのか？


def em_vj(self,src,seed_list,seed_img):
    
    dd = DisplayData()
    # フィルタの大きさは7とする
    # もっとも反応するフィルタは、岩よりも大きいはず
    # 11でやってる
    img = copy.deepcopy(src)
    img = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    img2= cv2.cvtColor(src,cv2.COLOR_BGR2GRAY) 
    
    seed = seed_list[80]

    x = seed[0]
    y = seed[1]
    print x,y

    img = img[y-10:y+10,x-10:x+10]
    img[10,10] = 0

    # 平均の準備,明，暗それぞれ, あと中間も定義する，そしてrg
    # もしくは明と暗をつなぐように中間？
    light_ave = (img2[y-1,x-1]*1. + img2[y,x-1] + img2[y+1,x-1])/3 # これだとfloatになる
    shade_ave = (img2[y-1,x+1]*1. + img2[y,x+1] + img2[y+1,x+1])/3 # これだとfloatになる
    # print light_ave,shade_ave

    ''' ガウシアン関数の用意 '''
    kernlen = 20
    nsig = 3
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    kernel = cv2.normalize(kernel, 0, 255, norm_type = cv2.NORM_MINMAX).astype(np.uint8)
    kernel = 255 - kernel
    kernel = kernel.astype(np.float)
    kernel = cv2.normalize(kernel, 0, 1, norm_type = cv2.NORM_MINMAX)

    # 平均と分散の用意、kernelの大きさが11だから、それに基づいて確実に岩っぽいとことる

    E = np.zeros_like(img)

    # エネルギー導出
    for i in range(20):
        for j in range(20):
            print i,j
            E[i,j] =  abs(img[i,j]-light_ave)*kernel[i,j]

    # 試しに見てみる
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    b,g,r = cv2.split(img)
    r0 = copy.deepcopy(r)
    g0 = copy.deepcopy(g)
    for i in range(1,100,10):
        print i
        r = r0
        g = g0
        r[E<i] = 255
        g[E<i] = 0
        img = cv2.merge((b,g,r))
        cv2.imshow('img',img)
        cv2.waitKey(0)
        
    return 0,0

def display_result(img,mask,format,color):
        '''
        imgの上にmaskを重ねて表示する
        img : 3ch, 512x512
        mask: 1ch, 512x512
        format: str, fill or edge
        '''
        # print type(img.dtype)
        if len(img.shape) == 2:
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        # colorだったらgrayに変換
        if len(mask.shape) == 3:
            img = img.astype(np.uint8)
            mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

        # dtypeがuint8じゃなければ変換
        if mask.dtype != 'uint8':
            mask = cv2.normalize(mask, 0, 255, norm_type = cv2.NORM_MINMAX)
            mask = mask.astype(np.uint8)

        ''' fill '''
        b,g,r = cv2.split(img)
        if color == 'r':
            r[mask != 0] = 255
            # g[mask != 0] = 0
            # b[mask != 0] = 0

        elif color == 'g':
            g[mask != 0] = 255
            r[mask != 0] = 0
            b[mask != 0] = 0

        else:
            b[mask != 0] = 255
        fill_result = cv2.merge((b,g,r))

        # エッジにする
        mask = cv2.Canny(mask, 0, 0,apertureSize = 3)
        b,g,r = cv2.split(img)
        if color == 'r':
            r[mask != 0] = 255
            # g[mask != 0] = 0
            # b[mask != 0] = 0        

        elif color == 'g':
            r[mask != 0] = 0
            g[mask != 0] = 255
            b[mask != 0] = 0  

        else:
            r[mask != 0] = 0
            g[mask != 0] = 0
            b[mask != 0] = 255              

        edge_result = cv2.merge((b,g,r))


        if format == 'fill':
            result = fill_result
        elif format == 'edge':
            result = edge_result
        else:
            result = img


        return result

def detect_maxima(src,thresh):
        '''
        閾値よりも大きい極大値を見つけて，imgとlistとして返す
        '''

        # colorだったらgrayに変換
        if len(src.shape) == 3:
            src=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        src_gray=np.asarray(copy.deepcopy(src))

        src_gray[src_gray<thresh] = 0 # 閾値処理

        gray_shifted = src_gray[0:src_gray.shape[0],0:-1:]
        zero_comp_ver = np.zeros((src_gray.shape[0],1))
        zero_comp_ver = zero_comp_ver.astype(np.int16)
        right_img= np.append(zero_comp_ver,gray_shifted,1)
        right_img= np.asarray(right_img)

        gray_shifted = src_gray[0:src_gray.shape[0],1::]
        left_img= np.append(gray_shifted,zero_comp_ver,1)
        left_img = np.asarray(left_img)

        gray_shifted = src_gray[1:src_gray.shape[0],0::]
        zero_comp_holi = np.zeros((1,src_gray.shape[1]))
        zero_comp_holi = zero_comp_holi.astype(np.int16)
        up_img = np.append(gray_shifted,zero_comp_holi,0)
        up_img = np.asarray(up_img)

        gray_shifted = src_gray[0:src_gray.shape[0]-1,0::]
        down_img = np.append(zero_comp_holi,gray_shifted,0)
        down_img = np.asarray(down_img)

        right_img = right_img.astype(np.int16)
        left_img = left_img.astype(np.int16)
        up_img = up_img.astype(np.int16)
        down_img = down_img.astype(np.int16)
        src_gray = src_gray.astype(np.int16)
        '''
        グレー元画像とシフト画像との差分
        '''
        sub_right_img=src_gray-right_img
        sub_left_img=src_gray-left_img
        sub_up_img=src_gray-up_img
        sub_down_img=src_gray-down_img

        k=0
        #Map=np.zeros((src_gray.shape[0]*src_gray.shape[1],2))
        '''
        返り値の用意
        '''
        Map=[]
        #print (sub_right_img.dtype)
        maps = np.zeros_like(src_gray)
        #print (maps.dtype)
        '''
        極値の探索
        line88,105,123,140のレンジの第２要素目の+1の必要性を吟味
        '''

        for i in range(0,src_gray.shape[0]):
            for j in range(0,src_gray.shape[1]):

                # 上下左右のすべてより大きい場合
                if sub_right_img[i,j]>0 and sub_left_img[i,j]>0 and sub_up_img[i,j]>0 and sub_down_img[i,j]>0:
                    Map.append([j,i])
                    maps[i,j] = 255
                    k=k+1

                # 横に続く場合
                elif(sub_right_img[i,j]==0) and (sub_left_img[i,j]>0) and (sub_up_img[i,j]>0) and (sub_down_img[i,j]>0) :
                    l=j-1

                    while (l>0):

                        if(sub_right_img[i,l]<0 or sub_up_img[i,l]<=0 or sub_down_img[i,l]<=0):

                            break
                        elif (sub_right_img[i,l]>0):
                            for r in range(l,j+1):
                                Map.append([i,r])
                                maps[r,i]=255
                                k=k+1
                            break
                        elif sub_right_img[i,l]==0:
                            l = l-1

                        else:
                            break

                # # 縦に続く場合
                elif sub_right_img[i,j]>0 and sub_left_img[i,j]>0 and sub_up_img[i,j]==0 and sub_down_img[i,j]>0:
                    l=i+1

                    while (l<src_gray.shape[0]):
                        if(sub_up_img[l,j]<0 or sub_right_img[l,j]<=0 or sub_left_img[l,j]<=0):
                            break
                        elif (sub_up_img[l,j]>0):
                            for r in range(i,l+1):
                                Map.append([j,r])
                                maps[r,j]=255
                                k=k+1
                            break
                        elif (sub_up_img[l,j]==0):
                            l=l+1
                        else:
                            break
                else:
                    continue

        maps = maps.astype(np.uint8)
        # print 'seed number =',k

        maxima_list = np.array(Map)

        return maps

def display_3D(img):
    '''
        入力画像を3Dで表示する
        args: 1ch image
    '''
    # データの準備
    x = np.arange(0, len(img[0]), 1)
    y = np.arange(0, len(img[1]), 1)
    X, Y = np.meshgrid(x, y) 
    Z = img

    # plot
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(X,Y,Z)

    # 設定
    ax.set_xlabel('pixel')
    ax.set_ylabel('pixel')        
    ax.set_zlabel('intensity')
    # ax.set_zlim(0, 300)
    ax.set_title('Image')
    ax.plot_surface(X, Y, Z, rstride=10, cstride=10, cmap = 'jet',linewidth=0)
    # ax.plot_wireframe(X,Y,Z, cmap = 'Greys', rstride=10, cstride=10)

    plt.pause(.001) # これだけでok
    # plt.show()
