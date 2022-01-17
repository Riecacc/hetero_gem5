embed_dim = [96,192,384,768]
depths = [2,2,18,2]
heads_num = [3,6,12,24]
window_size = 7
resolution = [224,224]
window_num = [1024,256,128,64]
patch_size = [resolution[0]/window_size,resolution[1]/window_size]
fc_ratio = 4

file = open('swintransformer.txt','w')
file.write("#layer_name, layer_type(CONV,POOL,FC), i_H, i_W, i_ch,\n")
file.write("#kernel_size, stride, padding, o_ch\n")
layer_num = 0
for i in range(len(depths)):
    #PQK的生成
    p1 = embed_dim[i]
    q1 = 3
    c1 = embed_dim[i]
    k1 = resolution[0]*resolution[1]
    r1 = 1
    s1 = 1  #s其实没用
    stride1 = 1
    head_dim = embed_dim[i]/heads_num[i]
    #A=KTQ
    p2 = patch_size[0]*patch_size[1]
    q2 = 1
    c2 = head_dim
    k2 = patch_size[0]*patch_size[1]
    r2 = 1
    s2 = 1
    stride2 = 1
    #O=VA
    p3 = embed_dim[i]
    q3 = 1
    c3 = patch_size[0]*patch_size[1]
    k3 = patch_size[0]*patch_size[1]
    r3 = 1
    s3 = 1
    stride3 = 1
    #O=WO
    p4 = embed_dim[i]
    q4 = 1
    c4 = embed_dim[i]
    k4 = resolution[0]*resolution[1]
    r4 = 1
    s4 = 1
    stride4 = 1
    #MLP
    p5 = fc_ratio*embed_dim[i]
    q5 = 1
    c5 = embed_dim[i]
    k5 = resolution[0]*resolution[1]/(4**(i-1))
    r5 = 1
    s5 = 1
    stride5 = 1
    for j in range(depths[i]):
        #PQK的生成
        file.write("layer%d CONV %d %d %d %d %d %d %d\n"%(layer_num,p1,q1,c1,r1,stride1,0,k1))
        layer_num = layer_num + 1
        #A=KTQ
        for k in range(window_num[i]*heads_num[i]):
            file.write("layer%d CONV %d %d %d %d %d %d %d\n"%(layer_num,p2,q2,c2,r2,stride2,0,k2))
            layer_num = layer_num + 1
        #O=VA
        for k in range(window_num[i]*heads_num[i]):
            file.write("layer%d CONV %d %d %d %d %d %d %d\n"%(layer_num,p3,q3,c3,r3,stride3,0,k3))
            layer_num = layer_num + 1
        #O=WO
        for k in range(window_num[i]):
            file.write("layer%d CONV %d %d %d %d %d %d %d\n"%(layer_num,p4,q4,c4,r4,stride4,0,k4))
            layer_num = layer_num + 1
        #MLP
        file.write("layer%d CONV %d %d %d %d %d %d %d\n"%(layer_num,p5,q5,c5,r5,stride5,0,k5))
        layer_num = layer_num + 1  
