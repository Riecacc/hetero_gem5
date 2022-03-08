# issues
# 1、AL1的需求量计算未考虑stride
# 4、repeat计算还是有问题，不能直接除
# 5、cp + 1越界问题
# 8、energy_die2die、energy_psum_list、delay_psum
# 9、task_file生成，最小循环
# 11、各链路通信需求计算需要并行度上的优化
# 12、A_W_offset是什么
# 13、F_cur和个体的编码未匹配上
# 14、scheduling, tiling, and mapping
# 15、注意numpy是浅拷贝
# 16、taskfile生成中pe对应的通信量部分为完成
import math
import random
import numpy as np
from config import *

#***********************************************************************#
# **********************************GA**********************************#
#***********************************************************************#

# for 与 存储资源的相关性
# 顺序LA P Q K C R S
A_correlation = np.array([1,1,1,0,1,1,1])
W_correlation = np.array([1,0,0,1,1,1,1])
O_correlation = np.array([1,1,1,1,0,0,0])

# 硬件参数
# mem
mem_minimum = 0 #辅助找cp点
AL1 = 1000 #bits
WL1 = 1000 
OL1 = 1000
AL2 = 10000
WL2 = 10000
OL2 = 10000
a_width = 8
w_width = 8
o_width = 32
A_W_offset = {"o":0, "a":5, "w":10}
# pe
pc1 = 8
pk1 = 8

class GA:
    def __init__(self, network_param, HW_param, population_order = 2 ,population_for = 2, debug = 0):
        self.HW_param = HW_param    #not include invalid node
        self.Chiplets_h = HW_param["Chiplet"][0]
        self.Chiplets_w = HW_param["Chiplet"][1]
        self.Chiplets = self.Chiplets_w * self.Chiplets_h
        self.PEs_h = HW_param["PE"][0]
        self.PEs_w = HW_param["PE"][1]
        self.PEs = self.PEs_w * self.PEs_h
        self.intra_PE = HW_param["intra_PE"]    #["C":4,"K":4]
        #self.parallel_level = parallel_level
        self.network_param = network_param
        self.size = {}
        self.getSize()          #每个维度的循环次数(考虑PK0和PC0)
        self.stride = network_param["stride"]
        self.P = self.size["P"]
        self.Q = self.size["Q"]
        self.C = math.ceil(self.size["C"]/pc1)
        self.K = math.ceil(self.size["K"]/pk1)
        self.R = self.size["R"]
        self.S = self.size["S"]
        self.LA = self.size["LA"]
        self.NoC_node_offset = []
        self.NoP2NoCnode = []
        #self.flag = flag
        #self.debug = debug
        #self.debug_file = debug_file
        #self.chiplet_parallel = chiplet_parallel
        #self.parallel_select, self.parallel_type = config_parallel_type(chiplet_parallel, core_parallel)    #并行模式
        self.population_for = population_for
        self.population_order = population_order
        # mapping方案
        self.pla3 = np.empty([1,population_for])
        self.pp3 = np.empty([1,population_for])
        self.pq3 = np.empty([1,population_for]) 
        self.pk3 = np.empty([1,population_for]) 
        self.pla2 = np.empty([1,population_for])
        self.pp2 = np.empty([1,population_for])
        self.pq2 = np.empty([1,population_for]) 
        self.pk2 = np.empty([1,population_for])
        self.la3 = np.empty([1,population_for])
        self.la2 = np.empty([1,population_for])
        self.la1 = np.empty([1,population_for])
        self.p3 = np.empty([1,population_for])
        self.p2 = np.empty([1,population_for])
        self.p1 = np.empty([1,population_for])
        self.q3 = np.empty([1,population_for])
        self.q2 = np.empty([1,population_for])
        self.q1 = np.empty([1,population_for])
        self.k3 = np.empty([1,population_for])
        self.k2 = np.empty([1,population_for])
        self.k1 = np.empty([1,population_for])
        self.c3 = np.empty([1,population_for])
        self.c2 = np.empty([1,population_for])
        self.c1 = np.empty([1,population_for])
        self.pe_order = np.arange(7)
        self.noc_order = np.arange(5)
        self.nop_order = np.arange(5)
        # 执行mapping方案初始化
        if debug == 1:
            self.ga_debug_init()
        else:
            self.ga_init()

    def getSize(self):
        for i in self.intra_PE:
            num = math.ceil(self.network_param[i] / self.intra_PE[i])
            self.size[i] = num
        for i in self.network_param:
            if i not in self.size:
                self.size[i] = self.network_param[i]


    def ga_init(self):
        #确定chiplet_level par_for
        #pla3 = random.randint(1,min(self.Chiplets,self.LA))
        #pp3 = random.randint(1,min(int(self.Chiplets/pla3),self.P))
        #pq3 = random.randint(1,min(int(self.Chiplets/pla3/pp3),self.Q))
        #pk3 = random.randint(1,min(int(self.Chiplets/pla3/pp3/pq3),self.K))
        self.pla3 = np.random.randint(1,min(self.Chiplets,self.LA)+1,size=(1,self.population_for))
        pp3_rand_max = self.Chiplets/self.pla3
        pp3_rand_max[pp3_rand_max > self.P] = self.P
        print(pp3_rand_max)
        self.pp3 = np.random.randint(1,pp3_rand_max+1,size=(1,self.population_for))
        pq3_rand_max = self.Chiplets/self.pla3/self.pp3
        pq3_rand_max[pq3_rand_max > self.Q] = self.Q
        print(pq3_rand_max)
        self.pq3 = np.random.randint(1,pq3_rand_max+1,size=(1,self.population_for))
        pk3_rand_max = self.Chiplets/self.pla3/self.pp3/self.pq3
        pk3_rand_max[pk3_rand_max > self.K] = self.K
        print(pk3_rand_max)
        self.pk3 = np.floor(pk3_rand_max)
        #self.pk3 = np.random.randint(1,pk3_rand_max+1,size=(1,self.population_for))
        print("----nop_level_parallel----")
        print("pla3:",self.pla3)
        print("pp3:",self.pp3)
        print("pq3:",self.pq3)
        print("pk3:",self.pk3)
        #确定pe_level par_for
        #pla2 = random.randint(1,min(self.PEs,int(self.LA/pla3)))
        #pp2 = random.randint(1,min(int(self.PEs/pla2),int(self.P/pp3)))
        #pq2 = random.randint(1,min(int(self.PEs/pla2/pp2),int(self.Q/pq3)))
        #pk2 = random.randint(1,min(int(self.PEs/pla2/pp2/pq2),int(self.K/pk3)))
        pla2_rand_max = self.LA/self.pla3
        pla2_rand_max[pla2_rand_max > self.PEs] = self.PEs
        self.pla2 = np.random.randint(1,pla2_rand_max+1,size=(1,self.population_for))
        pp2_rand_max = np.amin( np.stack([self.PEs/self.pla2,self.P/self.pp3]) , axis=0 )
        self.pp2 = np.random.randint(1,pp2_rand_max+1,size=(1,self.population_for))
        pq2_rand_max = np.amin( np.stack([self.PEs/self.pla2/self.pp2, self.Q/self.pq3]) , axis=0 )
        self.pq2 = np.random.randint(1,pq2_rand_max+1,size=(1,self.population_for))
        pk2_rand_max = np.amin( np.stack([self.PEs/self.pla2/self.pp2/self.pq2, self.K/self.pk3]) , axis=0 )
        self.pk2 = np.floor(pk2_rand_max)
        #self.pk2 = np.random.randint(1,pk2_rand_max+1,size=(1,self.population_for))
        print("----noc_level_parallel----")
        print("pla2:",self.pla2)
        print("pp2:",self.pp2)
        print("pq2:",self.pq2)
        print("pk2:",self.pk2)
        #确定LA,P,K,C
        self.la3 = np.random.randint(1,self.LA/self.pla3/self.pla2+1,size=(1,self.population_for))
        self.la2 = np.random.randint(1,self.LA/self.pla3/self.pla2/self.la3+1,size=(1,self.population_for))
        self.la1 = self.LA/self.pla3/self.pla2/self.la3/self.la2
        # self.la1 = np.random.randint(1,self.LA/self.pla3/self.pla2/self.la3/self.la2+1,size=(1,self.population_for))
        self.p3 = np.random.randint(1,self.P/self.pp3/self.pp2+1,size=(1,self.population_for))
        self.p2 = np.random.randint(1,self.P/self.pp3/self.pp2/self.p3+1,size=(1,self.population_for))
        self.p1 = np.ceil(self.Q/self.pq3/self.pq2)
        # self.p1 = np.random.randint(1,self.P/self.pp3/self.pp2/self.p3/self.p2+1,size=(1,self.population_for))
        self.q3 = np.random.randint(1,self.Q/self.pq3/self.pq2+1,size=(1,self.population_for))
        self.q2 = np.random.randint(1,self.Q/self.pq3/self.pq2/self.q3+1,size=(1,self.population_for))
        self.q1 = np.ceil(self.Q/self.pq3/self.pq2/self.q3/self.q2)
        # self.q1 = np.random.randint(1,self.Q/self.pq3/self.pq2/self.q3/self.q2+1,size=(1,self.population_for))
        self.k3 = np.random.randint(1,self.K/self.pk3/self.pk2+1,size=(1,self.population_for))
        self.k2 = np.random.randint(1,self.K/self.pk3/self.pk2/self.k3+1,size=(1,self.population_for))
        self.k1 = np.ceil(self.K/self.pk3/self.pk2/self.k3/self.k2)
        # self.k1 = np.random.randint(1,self.K/self.pk3/self.pk2/self.k3/self.k2+1,size=(1,self.population_for))
        self.c3 = np.random.randint(1,self.C+1,size=(1,self.population_for))
        self.c2 = np.random.randint(1,self.C/self.c3+1,size=(1,self.population_for))
        self.c1 = np.ceil(self.C/self.c3/self.c2)
        # self.c1 = np.random.randint(1,self.C/self.c3/self.c2+1,size=(1,self.population_for))
        self.r1 = self.R * np.ones(self.population_for)
        self.s1 = self.S * np.ones(self.population_for)
        #确定循环顺序:0~6分别对应LA P Q K C R S 
        #数组第0个元素表示对应FOR在最内层

        self.pe_order = np.arange(7)
        self.noc_order = np.arange(5)
        self.nop_order = np.arange(5)
        for i in range(self.population_order-1):
            self.pe_order = np.vstack([np.random.permutation(7),self.pe_order])
            self.noc_order = np.vstack([np.random.permutation(5),self.noc_order])
            self.nop_order = np.vstack([np.random.permutation(5),self.nop_order])

    def ga_debug_init(self):
        self.pla3 = np.array([1,1])
        self.pp3 = np.array([1,1])
        self.pq3 = np.array([1,1])
        self.pk3 = np.array([16,16])
        print("----nop_level_parallel----")
        print("pla3:",self.pla3[0])
        print("pp3:",self.pp3[0])
        print("pq3:",self.pq3[0])
        print("pk3:",self.pk3[0])
        self.pla2 = np.array([1,1])
        self.pp2 = np.array([1,1])
        self.pq2 = np.array([1,1])
        self.pk2 = np.array([1,1])
        print("----noc_level_parallel----")
        print("pla2:",self.pla2[0])
        print("pp2:",self.pp2[0])
        print("pq2:",self.pq2[0])
        print("pk2:",self.pk2[0])
        #确定LA,P,K,C
        self.la3 = np.array([1,1])
        self.la2 = np.array([1,1])
        self.la1 = np.array([1,1])

        self.p3 = np.array([1,1])
        self.p2 = np.array([20,20])
        self.p1 = np.array([2,2])

        self.q3 = np.array([1,1])
        self.q2 = np.array([2,2])
        self.q1 = np.array([21,21])

        self.k3 = np.array([1,1])
        self.k2 = np.array([1,1])
        self.k1 = np.array([1,1])

        self.c3 = np.array([1,1])
        self.c2 = np.array([1,1])
        self.c1 = np.array([1,1])

        self.r1 = np.array([5,5])
        self.s1 = np.array([5,5])
        #确定循环顺序:0~6分别对应LA P Q K C R S 
        #数组第0个元素表示对应FOR在最内层

        self.pe_order = np.array([[2,1,3,4,5,6,0],[2,1,3,4,5,6,0]])
        self.noc_order = np.array([[2,1,3,4,0],[2,1,3,4,0]])
        self.nop_order = np.array([[2,1,3,4,0],[2,1,3,4,0]])

    # 此处的num和w均指gem5仿真器中的拓扑图结构，包含没用的结点
    def construct_noc_Mesh(self,NOC_NODE_NUM, NoC_w):
        noc_dict = {}
        ol2_node_id = 0
        al2_node_id = NoC_w
        wl2_node_id = 2*NoC_w
        pe_node_list = []
        F = {} # fitness value for each link
        bw_scales = {}
        energy_ratio = {}
        # construct noc nodes 
        for src_local_id in range (NOC_NODE_NUM):  
            src = src_local_id
            local = src + 1000
            F[(local,src)] = np.zeros([self.population_order,self.population_for])
            F[(src,local)] = np.zeros([self.population_order,self.population_for])
            bw_scales[(local,src)] = 1
            bw_scales[(src,local)] = 1
            energy_ratio[(local,src)] = NOC_energy_ratio
            energy_ratio[(src,local)] = NOC_energy_ratio
            src_x = src_local_id %  NoC_w
            src_y = int(src_local_id / NoC_w)
            if src_x != 0:
                pe_node_list.append(src_local_id)
            for dst_local_id in range (NOC_NODE_NUM):    
                dst = dst_local_id            
                dst_x = dst_local_id %  NoC_w
                dst_y = int(dst_local_id / NoC_w)
                if (src_x == dst_x) :
                    if (src_y - dst_y == 1) or (src_y- dst_y == -1) :
                        F[(src,dst)] = np.zeros([self.population_order,self.population_for])
                        bw_scales[(src,dst)] = 1
                        energy_ratio[(src,dst)] = NOC_energy_ratio
                elif (src_y == dst_y) :
                    if (src_x - dst_x == 1) or (src_x - dst_x == -1):
                        F[(src,dst)] = np.zeros([self.population_order,self.population_for])
                        bw_scales[(src,dst)] = 1
                        energy_ratio[(src,dst)] = NOC_energy_ratio
    
        print ("----- finish construct the noc mesh ---- \n\n")

        noc_route_table = {}
        for src in range (0,NOC_NODE_NUM):
            for dst in range (0,NOC_NODE_NUM):
                noc_route_table[(src,dst)] = []
                cur_src = src
                cur_dst = src
                while cur_src != dst:
                    src_noc_x = cur_src  %  NoC_w
                    src_noc_y = int(cur_src / NoC_w)
                    dst_noc_x = dst %  NoC_w
                    dst_noc_y = int(dst / NoC_w)
                    if (src_noc_x > dst_noc_x): # go west
                        cur_noc_dst = src_noc_x-1 +  src_noc_y * NoC_w
                    elif (src_noc_x < dst_noc_x): # go east
                        cur_noc_dst = src_noc_x+1 +  src_noc_y * NoC_w
                    elif (src_noc_y < dst_noc_y): # go north
                        cur_noc_dst = src_noc_x + (src_noc_y+1) * NoC_w
                    elif (src_noc_y > dst_noc_y): # go south
                        cur_noc_dst = src_noc_x + (src_noc_y-1) * NoC_w
                    cur_dst = cur_noc_dst
                    noc_route_table[(src,dst)].append((cur_src,cur_dst))
                    cur_src = cur_dst
        noc_dict["route_table"] = noc_route_table
        noc_dict["F"] = F
        noc_dict["bw_scales"] = bw_scales
        noc_dict["energy_ratio"] = energy_ratio
        return noc_dict, al2_node_id, wl2_node_id, ol2_node_id, pe_node_list

    # 此处的num和w均指gem5仿真器中的拓扑图结构，包含没用的结点
    def construct_nop_Mesh(self,NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio, ddr_scale_ratio):
        nop_dict = {}
        dram_node_id = 0
        l2_node_list = []
        CORE_NUM = NOC_NODE_NUM*NOP_SIZE
        ALL_SIM_NODE_NUM = CORE_NUM + NOP_SIZE

        F = {} # fitness value for each link
        bw_scales = {}
        energy_ratio = {}
        node_list = []

        # construct NoP nodes
        for src_nop_id in range (NOP_SIZE):
            node_list.append(src_nop_id)
            src =  NOC_NODE_NUM * NOP_SIZE + src_nop_id
            local = src + 1000
            F[(local,src)] = np.zeros([self.population_order,self.population_for])
            F[(src,local)] = np.zeros([self.population_order,self.population_for])
            bw_scales[(local,src)] = nop_scale_ratio
            bw_scales[(src,local)] = nop_scale_ratio
            energy_ratio[(local,src)] = DIE2DIE_energy_ratio
            energy_ratio[(src,local)] = DIE2DIE_energy_ratio
            src_x = src_nop_id %  NoP_w
            src_y = int(src_nop_id / NoP_w)
            for dst_nop_id in range (NOP_SIZE):
                dst =  NOC_NODE_NUM * NOP_SIZE + dst_nop_id
                dst_x = dst_nop_id %  NoP_w
                dst_y = int(dst_nop_id / NoP_w)
                if (src_x == dst_x) :
                    if (src_y - dst_y == 1) or (src_y- dst_y == -1) :
                        F[(src,dst)] = np.zeros([self.population_order,self.population_for])
                        bw_scales[(src,dst)] = nop_scale_ratio
                        energy_ratio[(src,dst)] = DIE2DIE_energy_ratio
                        
                elif (src_y == dst_y) :
                    if (src_x - dst_x == 1) or (src_x - dst_x == -1):
                        F[(src,dst)] = np.zeros([self.population_order,self.population_for])
                        bw_scales[(src,dst)] = nop_scale_ratio
                        energy_ratio[(src,dst)] = DIE2DIE_energy_ratio

        # construct noc and nop connection
        for nop_id in range (1,NOP_SIZE):
            nop_router_id = nop_id + NOC_NODE_NUM * NOP_SIZE 
            noc_router_id = nop_id * NOC_NODE_NUM
            if (noc_router_id%(NOC_NODE_NUM*NoP_w) != 0):
                l2_node_list.append(noc_router_id)
            node_list.append(noc_router_id)
            F[(noc_router_id,nop_router_id)] = np.zeros([self.population_order,self.population_for])
            F[(nop_router_id,noc_router_id)] = np.zeros([self.population_order,self.population_for])
            bw_scales[(noc_router_id,nop_router_id)] = nop_scale_ratio
            bw_scales[(nop_router_id,noc_router_id)] = nop_scale_ratio
            energy_ratio[(noc_router_id,nop_router_id)] = DIE2DIE_energy_ratio
            energy_ratio[(nop_router_id,noc_router_id)] = DIE2DIE_energy_ratio
        # construct ddr and nop connection
        nop_router_id = 0 + NOC_NODE_NUM * NOP_SIZE 
        noc_router_id = 0 * NOC_NODE_NUM
        node_list.append(noc_router_id)
        F[(noc_router_id,nop_router_id)] = np.zeros([self.population_order,self.population_for])
        F[(nop_router_id,noc_router_id)] = np.zeros([self.population_order,self.population_for])
        bw_scales[(noc_router_id,nop_router_id)] = ddr_scale_ratio
        bw_scales[(nop_router_id,noc_router_id)] = ddr_scale_ratio
        energy_ratio[(noc_router_id,nop_router_id)] = DIE2DIE_energy_ratio
        energy_ratio[(nop_router_id,noc_router_id)] = DIE2DIE_energy_ratio
        print ("----- finish construct the nop mesh ---- \n\n")

        nop_route_table = {}
        hops = {}

        def noc_id (real_id):
            return (real_id % NOC_NODE_NUM)

        def chip_id (real_id):
            return (int (real_id /NOC_NODE_NUM) )

        def comm_id (real_id): # the communication router id in one chip
            return (chip_id(real_id)*NOC_NODE_NUM)

        def nop_id (real_id):
            return (real_id - NOC_NODE_NUM*NOP_SIZE)
        
        for src in node_list:
            for dst in node_list:
                # print ("src = ",src,"dst = ",dst)
                nop_route_table[(src,dst)] = []
                cur_src = src
                cur_dst = src
                if chip_id(cur_src) != chip_id(dst):
                    while cur_src != comm_id(src): # go to the first noc node
                        src_noc_x = noc_id(cur_src) %  NoC_w
                        src_noc_y = int(noc_id(cur_src) / NoC_w)
                        dst_noc_x = noc_id(comm_id(src)) % NoC_w
                        dst_noc_y = int(noc_id(comm_id(src))/ NoC_w)
                        # print (comm_id(src),src_noc_x,src_noc_y,dst_noc_x,dst_noc_y)
                        if (src_noc_x > dst_noc_x):  # go west
                            cur_noc_dst = src_noc_x-1 +  src_noc_y * NoC_w
                        elif (src_noc_x < dst_noc_x): # go east
                            cur_noc_dst = src_noc_x+1 +  src_noc_y * NoC_w
                        elif (src_noc_y < dst_noc_y): # go north
                            cur_noc_dst = src_noc_x + (src_noc_y+1) * NoC_w
                        elif (src_noc_y > dst_noc_y): # go south
                            cur_noc_dst = src_noc_x + (src_noc_y-1) * NoC_w
                        cur_dst = chip_id(cur_src) * NOC_NODE_NUM + cur_noc_dst
                        nop_route_table[(src,dst)].append((cur_src,cur_dst))
                        cur_src = cur_dst
                        
                    # go to the nop node
                    cur_dst = chip_id(cur_src) + NOC_NODE_NUM * NOP_SIZE
                    nop_route_table[(src,dst)].append((cur_src,cur_dst))
                    cur_src = cur_dst

                    while cur_src != chip_id(dst) + NOC_NODE_NUM * NOP_SIZE : # nop router of the destination node
                        src_nop_x = nop_id(cur_src) % NoP_w
                        src_nop_y = int (nop_id(cur_src) / NoP_w)
                        dst_nop_x = chip_id(dst) % NoP_w
                        dst_nop_y = int(chip_id(dst) / NoP_w)
                        if (src_nop_x > dst_nop_x):  # go west
                            cur_nop_dst = src_nop_x-1 +  src_nop_y * NoP_w
                        elif (src_nop_x < dst_nop_x): # go east
                            cur_nop_dst = src_nop_x+1 +  src_nop_y * NoP_w
                        elif (src_nop_y < dst_nop_y): # go north
                            cur_nop_dst = src_nop_x + (src_nop_y+1) * NoP_w
                        elif (src_nop_y > dst_nop_y): # go south
                            cur_nop_dst = src_nop_x + (src_nop_y-1) * NoP_w
                        cur_dst = cur_nop_dst+NOC_NODE_NUM*NOP_SIZE
                        nop_route_table[(src,dst)].append((cur_src,cur_dst))
                        cur_src = cur_dst
                    
                    # go to the communication id 
                    cur_dst = chip_id(dst) * NOC_NODE_NUM
                    nop_route_table[(src,dst)].append((cur_src,cur_dst))
                    cur_src = cur_dst

                while cur_src != dst:
                    src_noc_x = noc_id(cur_src)  %  NoC_w
                    src_noc_y = int(noc_id(cur_src)  / NoC_w)
                    dst_noc_x = noc_id(dst) %  NoC_w
                    dst_noc_y = int(noc_id(dst)  / NoC_w)
                    # print ("src_x",src_x,"src_y",src_y,"dst_x",dst_x,"dst_y",dst_y)
                    if (src_noc_x > dst_noc_x): # go west
                        cur_noc_dst = src_noc_x-1 +  src_noc_y * NoC_w
                    elif (src_noc_x < dst_noc_x): # go east
                        cur_noc_dst = src_noc_x+1 +  src_noc_y * NoC_w
                    elif (src_noc_y < dst_noc_y): # go north
                        cur_noc_dst = src_noc_x + (src_noc_y+1) * NoC_w
                    elif (src_noc_y > dst_noc_y): # go south
                        cur_noc_dst = src_noc_x + (src_noc_y-1) * NoC_w
                    cur_dst = chip_id(cur_src) * NOC_NODE_NUM + cur_noc_dst
                    nop_route_table[(src,dst)].append((cur_src,cur_dst))
                    cur_src = cur_dst

        nop_dict["route_table"] = nop_route_table
        nop_dict["F"] = F
        nop_dict["bw_scales"] = bw_scales
        nop_dict["energy_ratio"] = energy_ratio
        return nop_dict, dram_node_id, l2_node_list

    def evaluate(self):
        # 各层次理想状况存储需求
        # pe_level: la1 p1 q1 k1 c1 r1 s1 (pc1 pk1) 
        # noc_level: la2 p2 q2 k2 c2  (pla2 pp2 pq2 pk2)
        # nop_level: la3 p3 q3 k3 c3  (pla3 pp3 pq3 pk3)
        pe_serial_for = np.vstack([self.la1,self.p1,self.q1,self.k1,self.c1,self.r1,self.s1])
        noc_serial_for = np.vstack([self.la2,self.p2,self.q2,self.k2,self.c2])
        nop_serial_for = np.vstack([self.la3,self.p3,self.q3,self.k3,self.c3])
        computation_cycles = np.prod(noc_serial_for,axis=0) * np.prod(nop_serial_for,axis=0)\
            * np.prod(pe_serial_for,axis=0)
        # AL1,WL1,OL1 -> one of REGs 
        # 计算一次卷积一个PE需要的数据量
        mem_reg_a = pc1 * a_width
        mem_reg_w = pc1 * pk1 * w_width
        mem_reg_o = pk1 * o_width
        al1_to_reg = np.zeros([self.population_order,self.population_for])
        wl1_to_reg = np.zeros([self.population_order,self.population_for])
        ol1_to_reg = np.zeros([self.population_order,self.population_for])
        pe_serial_for_reorder = []
        for i in range(len(self.pe_order)):
            pe_serial_for_reorder.append(np.vstack([pe_serial_for[self.pe_order[i]],np.ones(self.population_for)])) #最后补充一行1是为了解决cp+1越界的问题
            # AL1 -> one of REGs
            # 找到第一个与A有关的FOR
            A_correlation_reorder = A_correlation[self.pe_order[i]]
            inner_for_reg_a = np.argwhere(A_correlation_reorder==1)
            reg_a_repeat = np.prod(pe_serial_for_reorder[i][inner_for_reg_a[0][0]:self.population_for+1,:],axis=0)\
                * np.prod(noc_serial_for,axis=0) * np.prod(nop_serial_for,axis=0)
            al1_to_reg[i] = mem_reg_a * reg_a_repeat
            # WL1 -> one of REGs
            W_correlation_reorder = W_correlation[self.pe_order[i]]
            inner_for_reg_w = np.argwhere(W_correlation_reorder==1)
            reg_w_repeat = np.prod(pe_serial_for_reorder[i][inner_for_reg_w[0][0]:self.population_for+1,:],axis=0)\
                * np.prod(noc_serial_for,axis=0) * np.prod(nop_serial_for,axis=0)
            wl1_to_reg[i] = mem_reg_w * reg_w_repeat
            # OL1 -> one of REGs
            O_correlation_reorder = O_correlation[self.pe_order[i]]
            inner_for_reg_o = np.argwhere(O_correlation_reorder==1)
            reg_o_repeat = np.prod(pe_serial_for_reorder[i][inner_for_reg_o[0][0]:self.population_for+1,:],axis=0)\
                * np.prod(noc_serial_for,axis=0) * np.prod(nop_serial_for,axis=0)
            ol1_to_reg[i] = mem_reg_o * reg_o_repeat


        # L2 -> L1
        """al1_repeat = []
        wl1_repeat = []
        ol1_repeat = []"""
        al2_to_al1 = np.zeros([self.population_order,self.population_for])
        wl2_to_wl1 = np.zeros([self.population_order,self.population_for])
        ol2_to_ol1 = np.zeros([self.population_order,self.population_for])
        # 利用率
        al1_utilization_ratio = np.zeros([self.population_order,self.population_for])
        wl1_utilization_ratio = np.zeros([self.population_order,self.population_for])
        ol1_utilization_ratio = np.zeros([self.population_order,self.population_for])
        for i in range(len(self.pe_order)):
            ## AL1
            # 考虑相关性
            al1_for = pe_serial_for * (np.expand_dims(A_correlation,axis=0).repeat(self.population_for,axis=0).T)
            al1_for[al1_for <= 0] = 1 
            # 按照顺序重新排列
            al1_for = al1_for[self.pe_order[i]] 
            # 找到cp点
            al1_need = np.cumprod(al1_for,axis=0) * pc1 * a_width 
                #* self.pp2 * self.pq2 * self.pla2
            al1_need[al1_need > AL1] = mem_minimum
            al1_cp = al1_need.shape[0] - 1 - np.argmax(np.flip(al1_need,axis=0),axis=0) #argmax找的是index最小的max所以要翻转一下
            # 计算L2->L1,cp点(含)以内考虑相关性,其他不考虑
            al2_to_al1_tmp = np.ones(self.population_for)
            for j in range(len(al1_cp)):
                al2_to_al1_tmp[j] = al1_need[al1_cp[j]][j] * np.prod(pe_serial_for_reorder[i][al1_cp[j]+1:,j])
            al2_to_al1[i] = al2_to_al1_tmp * np.prod(noc_serial_for,axis=0) * np.prod(nop_serial_for,axis=0)
            al1_utilization_ratio[i] = al1_need[al1_cp,list(range(self.population_for))] / AL1
            """# 计算重复读取次数: 仅考虑到pe_level的for带来的影响,随后需要在AL2时补充考虑noc,nop_level的影响
            # al1_repeat[i] = np.ceil(np.amin(al1_need,0)/AL1,dtype=int)
            al1_repeat[i] = np.amin(al1_need,0)/AL1
            al1_need[al1_cp,np.arange(self.population_for)] = mem_minimum
            al1_for[al1_need == mem_minimum] = 1
            al1_repeat[i] = al1_repeat[i] * np.prod(al1_for,axis=0)"""
            ## WL1
            wl1_for = pe_serial_for * (np.expand_dims(W_correlation,axis=0).repeat(self.population_for,axis=0).T)
            wl1_for[wl1_for <= 0] = 1 
            wl1_for = wl1_for[self.pe_order[i]] 
            wl1_need = np.cumprod(wl1_for,axis=0) * pc1 * pk1 * w_width 
                #* self.pk2 * self.pla2
            wl1_need[wl1_need > WL1] = mem_minimum
            wl1_cp = wl1_need.shape[0] - 1 - np.argmin(np.flip(wl1_need,axis=0),axis=0)
            # 计算L2->L1,cp点(含)以内考虑相关性,其他不考虑
            wl2_to_wl1_tmp = np.ones(self.population_for)
            for j in range(len(wl1_cp)):
                wl2_to_wl1_tmp[j] = wl1_need[wl1_cp[j]][j] * np.prod(pe_serial_for_reorder[i][wl1_cp[j]+1:,j])
            wl2_to_wl1[i] = wl2_to_wl1_tmp * np.prod(noc_serial_for,axis=0) * np.prod(nop_serial_for,axis=0)
            wl1_utilization_ratio[i] = wl1_need[wl1_cp,list(range(self.population_for))] / WL1
            """#wl1_repeat[i] = np.ceil(np.amin(wl1_need,0)/WL1,dtype=int)
            wl1_repeat[i] = np.amin(wl1_need,0)/WL1
            wl1_need[wl1_cp,np.arange(self.population_for)] = mem_minimum
            wl1_for[wl1_need == mem_minimum] = 1
            wl1_repeat[i] = wl1_repeat[i] * np.prod(wl1_for,axis=0)"""
            ## OL1
            ol1_for = pe_serial_for * (np.expand_dims(O_correlation,axis=0).repeat(self.population_for,axis=0).T)
            ol1_for[ol1_for <= 0] = 1 
            ol1_for = ol1_for[self.pe_order[i]] 
            ol1_need = np.cumprod(ol1_for,axis=0) * pk1 * o_width 
                #* self.pp2 * self.pq2 * self.pk2 * self.pla2
            ol1_need[ol1_need > OL1] = mem_minimum
            print("np.argmax")
            print(np.argmax(np.flip(ol1_need,axis=0),axis=0))
            ol1_cp = ol1_need.shape[0] - 1 - np.argmax(np.flip(ol1_need,axis=0),axis=0)
            # 计算L2->L1,cp点(含)以内考虑相关性,其他不考虑
            ol2_to_ol1_tmp = np.ones(self.population_for)
            for j in range(len(ol1_cp)):
                ol2_to_ol1_tmp[j] = ol1_need[ol1_cp[j]][j] * np.prod(pe_serial_for_reorder[i][ol1_cp[j]+1:,j])
            ol2_to_ol1[i] = ol2_to_ol1_tmp * np.prod(noc_serial_for,axis=0) * np.prod(nop_serial_for,axis=0)
            ol1_utilization_ratio[i] = ol1_need[ol1_cp,list(range(self.population_for))] / OL1
            """#ol1_repeat[i] = np.ceil(np.amin(ol1_need,0)/OL1,dtype=int)
            ol1_repeat[i] = np.amin(ol1_need,0)/OL1
            ol1_need[ol1_cp,np.arange(self.population_for)] = mem_minimum
            ol1_for[ol1_need == mem_minimum] = 1
            ol1_repeat[i] = ol1_repeat[i] * np.prod(ol1_for,axis=0)"""

        # DDR -> L2 
        ddr_to_al2 = np.zeros([self.population_order,self.population_for])
        ddr_to_wl2 = np.zeros([self.population_order,self.population_for])
        ddr_to_ol2 = np.zeros([self.population_order,self.population_for])
        l2_serial_for = np.vstack([pe_serial_for,noc_serial_for])
        l2_order = np.hstack([self.pe_order,self.noc_order])
        l2_serial_for_reorder = []
        l2_A_correlation = np.hstack([A_correlation,A_correlation[0:5]])
        l2_W_correlation = np.hstack([W_correlation,W_correlation[0:5]])
        l2_O_correlation = np.hstack([O_correlation,O_correlation[0:5]])
        # 利用率
        ol2_utilization_ratio = np.zeros([self.population_order,self.population_for])
        al2_utilization_ratio = np.zeros([self.population_order,self.population_for])
        wl2_utilization_ratio = np.zeros([self.population_order,self.population_for])
        for i in range(len(l2_order)):
            ## AL2
            # 考虑相关性
            al2_for = l2_serial_for * (np.expand_dims(l2_A_correlation,axis=0).repeat(self.population_for,axis=0).T)
            al2_for[al2_for <= 0] = 1 
            # 按照顺序重新排列
            al2_for = al2_for[l2_order[i]] 
            l2_serial_for_reorder.append(l2_serial_for[l2_order[i]])
            # 找到cp点
            al2_need = np.cumprod(al2_for,axis=0) * pc1 * a_width \
                * self.pp2 * self.pq2 * self.pla2 #* self.pp3 * self.pq3 * self.pla3 
            al2_need[al2_need > AL2] = mem_minimum
            al2_cp = al2_need.shape[0] - 1 - np.argmax(np.flip(al2_need,axis=0),axis=0)
            # 计算ddr->L2,cp点(含)以内考虑相关性,其他不考虑
            ddr_to_al2_tmp = np.ones(self.population_for)
            for j in range(len(al2_cp)):
                ddr_to_al2_tmp[j] = al2_need[al2_cp[j]][j] * np.prod(l2_serial_for_reorder[i][al2_cp[j]+1:,j])
            ddr_to_al2[i] = ddr_to_al2_tmp * np.prod(nop_serial_for,axis=0)
            al2_utilization_ratio[i] = al2_need[al2_cp,list(range(self.population_for))] / AL2
            ## WL2
            wl2_for = l2_serial_for * (np.expand_dims(l2_W_correlation,axis=0).repeat(self.population_for,axis=0).T)
            wl2_for[wl2_for <= 0] = 1 
            wl2_for = wl2_for[l2_order[i]] 
            wl2_need = np.cumprod(wl2_for,axis=0) * pc1 * pk1 * w_width \
                * self.pk2 * self.pla2 #* self.pk3 * self.pla3
            wl2_need[wl2_need > WL2] = mem_minimum
            wl2_cp = wl2_need.shape[0] - 1 - np.argmax(np.flip(wl2_need,axis=0),axis=0)
            # 计算ddr->L2,cp点(含)以内考虑相关性,其他不考虑
            ddr_to_wl2_tmp = np.ones(self.population_for)
            for j in range(len(wl2_cp)):
                ddr_to_wl2_tmp[j] = wl2_need[wl2_cp[j]][j] * np.prod(l2_serial_for_reorder[i][wl2_cp[j]+1:,j])
            ddr_to_wl2[i] = ddr_to_wl2_tmp * np.prod(nop_serial_for,axis=0)
            wl2_utilization_ratio[i] = wl2_need[wl2_cp,list(range(self.population_for))] / WL2
            ## OL2
            ol2_for = l2_serial_for * (np.expand_dims(l2_O_correlation,axis=0).repeat(self.population_for,axis=0).T)
            ol2_for[ol2_for <= 0] = 1 
            ol2_for = ol2_for[l2_order[i]] 
            ol2_need = np.cumprod(ol2_for,axis=0) * pk1 * o_width \
                * self.pp2 * self.pq2 * self.pk2 * self.pla2 #* self.pp3 * self.pq3 * self.pk3 * self.pla3
            print("ol2_need")
            print(ol2_need)
            ol2_need[ol2_need > OL2] = mem_minimum
            print("ol2_need_cp")
            print(ol2_need)
            ol2_cp = ol2_need.shape[0] - 1 - np.argmax(np.flip(ol2_need,axis=0),axis=0)
            print("ol2_cp")
            print(ol2_cp)
            # 计算ddr->L2,cp点(不含)以内考虑相关性,其他不考虑
            ddr_to_ol2_tmp = np.ones(self.population_for)
            for j in range(len(ol2_cp)):
                ddr_to_ol2_tmp[j] = ol2_need[ol2_cp[j]][j] * np.prod(l2_serial_for_reorder[i][ol2_cp[j]+1:,j])
            ddr_to_ol2[i] = ddr_to_ol2_tmp * np.prod(nop_serial_for,axis=0)
            ol2_utilization_ratio[i] = ol2_need[ol2_cp,list(range(self.population_for))] / OL2
        # print
        """print("ddr_to_al2:",ddr_to_al2)
        print("ddr_to_wl2:",ddr_to_wl2)
        print("ddr_to_ol2:",ddr_to_ol2)
        print("al2_to_al1:",al2_to_al1)
        print("wl2_to_wl1:",wl2_to_wl1)
        print("ol2_to_ol1:",ol2_to_ol1)
        print("ol2_utilization_ratio:",ol2_utilization_ratio)
        print("al2_utilization_ratio:",al2_utilization_ratio)
        print("wl2_utilization_ratio:",wl2_utilization_ratio)
        print("al1_utilization_ratio:",al1_utilization_ratio)
        print("wl1_utilization_ratio:",wl1_utilization_ratio)
        print("ol1_utilization_ratio:",ol1_utilization_ratio)"""
        # 各链路通信量计算
        # 构建拓扑图
        noc_dict, al2_node_id, wl2_node_id, ol2_node_id, pe_node_list = self.construct_noc_Mesh(self.PEs_h*(self.PEs_w+1), self.PEs_w+1)
        F_cur_noc = noc_dict["F"].copy() 
        nop_dict, dram_node_id, l2_node_list = self.construct_nop_Mesh(self.PEs_h*(self.PEs_w+1), self.PEs_w+1,\
            self.Chiplets_h*(self.Chiplets_w+1), self.Chiplets_w+1, nop_bandwidth/noc_bandwidth, ddr_bandwidth/noc_bandwidth)
        F_cur_nop = nop_dict["F"].copy()
        # ouput:l2->pe
        for dst in pe_node_list:
            for link in noc_dict["route_table"][ol2_node_id, dst]:
                F_cur_noc[link] += (ol2_to_ol1/computation_cycles/noc_dict["bw_scales"][link])
        # output:pe->l2
        for src in pe_node_list:
            for link in noc_dict["route_table"][src, ol2_node_id]:
                F_cur_noc[link] += (ol2_to_ol1/computation_cycles/noc_dict["bw_scales"][link])
        # weight:l2->pe
        for dst in pe_node_list:
            for link in noc_dict["route_table"][wl2_node_id, dst]:
                F_cur_noc[link] += (wl2_to_wl1/computation_cycles/noc_dict["bw_scales"][link])
        # activaiton:l2->pe
        for dst in pe_node_list:
            for link in noc_dict["route_table"][al2_node_id, dst]:
                F_cur_noc[link] += (al2_to_al1/computation_cycles/noc_dict["bw_scales"][link])
        # output:ddr->l2
        for dst in l2_node_list:
            F_cur_nop[(0, 1440)] = 1
            for link in nop_dict["route_table"][dram_node_id, dst]:
                F_cur_nop[link] += (ol2_to_ol1/computation_cycles/nop_dict["bw_scales"][link])
        # output:l2->ddr
        for src in l2_node_list:
            for link in nop_dict["route_table"][src, dram_node_id]:
                F_cur_nop[link] += (ol2_to_ol1/computation_cycles/nop_dict["bw_scales"][link])
        # weight:ddr->l2
        for dst in l2_node_list:
            for link in nop_dict["route_table"][dram_node_id, dst]:
                F_cur_nop[link] += (wl2_to_wl1/computation_cycles/nop_dict["bw_scales"][link])
        # activaiton:ddr->l2
        for dst in l2_node_list:
            for link in nop_dict["route_table"][dram_node_id, dst]:
                F_cur_nop[link] += (al2_to_al1/computation_cycles/nop_dict["bw_scales"][link])
        """degrade_ratio_nop = np.amax(np.array(list(F_cur_nop.values())),axis=0)
        degrade_ratio_noc = np.amax(np.array(list(F_cur_noc.values())),axis=0)
        degrade_ratio = degrade_ratio_noc
        degrade_ratio[degrade_ratio < degrade_ratio_nop] = degrade_ratio_nop[degrade_ratio < degrade_ratio_nop]"""
        degrade_ratio = np.amax(np.array(list(F_cur_noc.values())),axis=0)
        degrade_ratio[degrade_ratio < 1] = 1
        # 结果
        energy_rd_act_L1 = al1_to_reg*SRAM_energy(AL1)
        energy_rd_wgt_L1 = wl1_to_reg*SRAM_energy(WL1)
        energy_wr_opt_L1 = ol1_to_reg*SRAM_energy(OL1)
        energy_wr_opt_L2 = ol2_to_ol1*SRAM_energy(OL2)
        energy_rd_opt_L2 = ol2_to_ol1*SRAM_energy(OL2)
        energy_rd_wgt_L2 = wl2_to_wl1*SRAM_energy(WL2)
        energy_rd_act_L2 = al2_to_al1*SRAM_energy(AL2)
        energy_wr_opt_dram = ddr_to_ol2*DRAM_energy_ratio
        energy_rd_opt_dram = ddr_to_ol2*DRAM_energy_ratio
        energy_rd_wgt_dram = ddr_to_wl2*DRAM_energy_ratio
        energy_rd_act_dram = ddr_to_al2*DRAM_energy_ratio
        print("L1 energy:")
        print("al1rd:",energy_rd_act_L1[0][0],"  wl1rd:",energy_rd_wgt_L1[0][0],"  ol1:",energy_wr_opt_L1[0][0])
        print("L2 energy:")
        print("al2rd:",energy_rd_act_L2[0][0],"  wl2rd:",energy_rd_wgt_L2[0][0],"  ol2:",energy_wr_opt_L2[0][0])
        print("dram energy:")
        print("dram_a_rd:",energy_rd_act_dram[0][0],"  dram_w_rd:",energy_rd_wgt_dram[0][0],"  dram_o:",energy_rd_opt_dram[0][0])
        delay = degrade_ratio*computation_cycles
        print("delay:",delay)
        print("computation_cycles",computation_cycles)
        energy_L1_list = [energy_rd_wgt_L1, energy_rd_act_L1, energy_wr_opt_L1]
        energy_L2_list = [energy_wr_opt_L2, energy_rd_opt_L2, energy_rd_wgt_L2, energy_rd_act_L2]
        energy_dram_list = [energy_wr_opt_dram, energy_rd_opt_dram, energy_rd_wgt_dram, energy_rd_act_dram]
        energy_die2die = 0;	energy_core2core = 0

    def taskfile(self,act_wgt_dict,out_dict,multicast):
        # 各层次理想状况存储需求
        # pe_level: la1 p1 q1 k1 c1 r1 s1 (pc1 pk1) 
        # noc_level: la2 p2 q2 k2 c2  (pla2 pp2 pq2 pk2)
        # nop_level: la3 p3 q3 k3 c3  (pla3 pp3 pq3 pk3)
        pe_serial_for = np.vstack([self.la1,self.p1,self.q1,self.k1,self.c1,self.r1,self.s1])
        noc_serial_for = np.vstack([self.la2,self.p2,self.q2,self.k2,self.c2])
        nop_serial_for = np.vstack([self.la3,self.p3,self.q3,self.k3,self.c3])
        computation_cycles = np.prod(noc_serial_for,axis=0) * np.prod(nop_serial_for,axis=0)\
            * np.prod(pe_serial_for,axis=0)
        # AL1,WL1,OL1 -> one of REGs 
        # 计算一次卷积一个PE需要的数据量
        mem_reg_a = pc1 * a_width
        mem_reg_w = pc1 * pk1 * w_width
        mem_reg_o = pk1 * o_width
        al1_to_reg = []
        wl1_to_reg = []
        ol1_to_reg = []
        pe_serial_for_reorder = []            

        # L2 -> L1
        al2_to_al1 = []
        wl2_to_wl1 = []
        ol2_to_ol1 = []
        al1_cp = np.zeros([self.population_order,self.population_for])
        wl1_cp = np.zeros([self.population_order,self.population_for])
        ol1_cp = np.zeros([self.population_order,self.population_for])
        # 每个PE一次最小循环需要传输的数据
        pe_a_data = np.zeros([self.population_order,self.population_for])
        pe_w_data = np.zeros([self.population_order,self.population_for])
        pe_o_data = np.zeros([self.population_order,self.population_for])
        # cp点与最小循环的循环次数之比
        acp_pe_ratio = np.zeros(self.population_for)
        wcp_pe_ratio = np.zeros(self.population_for)
        ocp_pe_ratio = np.zeros(self.population_for)

        # DDR -> L2 
        ddr_to_al2 = np.zeros([self.population_order,self.population_for])
        ddr_to_wl2 = np.zeros([self.population_order,self.population_for])
        ddr_to_ol2 = np.zeros([self.population_order,self.population_for])
        l2_serial_for = np.vstack([pe_serial_for,noc_serial_for])
        l2_order = np.hstack([self.pe_order,self.noc_order])
        l2_serial_for_reorder = []
        l2_A_correlation = np.hstack([A_correlation,A_correlation[0:5]])
        l2_W_correlation = np.hstack([W_correlation,W_correlation[0:5]])
        l2_O_correlation = np.hstack([O_correlation,O_correlation[0:5]])
        al2_cp = np.zeros([self.population_order,self.population_for])
        wl2_cp = np.zeros([self.population_order,self.population_for])
        ol2_cp = np.zeros([self.population_order,self.population_for])
        # 每个chiplet一次最小循环需要传输的数据
        core_a_data = np.zeros([self.population_order,self.population_for])
        core_w_data = np.zeros([self.population_order,self.population_for])
        core_o_data = np.zeros([self.population_order,self.population_for])
        # cp点与最小循环的循环次数之比
        acp_core_ratio = np.zeros(self.population_for)
        wcp_core_ratio = np.zeros(self.population_for)
        ocp_core_ratio = np.zeros(self.population_for)
        # 最小循环计算cycle数
        core_computation = np.zeros([self.population_order,self.population_for])

        for i in range(len(self.pe_order)):
            ## L2->L1
            pe_serial_for_reorder[i] = np.vstack([pe_serial_for[self.pe_order[i]],np.ones(self.population_for)]) #最后补充一行1是为了解决cp+1越界的问题
            ## AL1
            # 考虑相关性
            al1_for = pe_serial_for * (np.expand_dims(A_correlation,axis=0).repeat(self.population_for,axis=0).T)
            al1_for[al1_for <= 0] = 1 
            # 按照顺序重新排列
            al1_for = al1_for[self.pe_order[i]] 
            # 找到cp点
            al1_need = np.cumprod(al1_for,axis=0) * pc1 * a_width 
                #* self.pp2 * self.pq2 * self.pla2
            al1_need[al1_need > AL1] = mem_minimum
            al1_cp[i] = al1_need.shape[0] - 1 - np.argmax(np.flip(al1_need,axis=0),axis=0) #argmax找的是index最小的max所以要翻转一下
            # 计算L2->L1,cp点(含)以内考虑相关性,其他不考虑
            al2_to_al1_tmp = np.ones(self.population_for)
            for j in al1_cp.shape[1]:
                al2_to_al1_tmp[j] = al1_need[al1_cp[i][j]][j] * np.prod(pe_serial_for_reorder[i][al1_cp[i][j]+1:,j])
            al2_to_al1[i] = al2_to_al1_tmp * np.prod(noc_serial_for,axis=0) * np.prod(nop_serial_for,axis=0)
            ## WL1
            wl1_for = pe_serial_for * (np.expand_dims(W_correlation,axis=0).repeat(self.population_for,axis=0).T)
            wl1_for[wl1_for <= 0] = 1 
            wl1_for = wl1_for[self.pe_order[i]] 
            wl1_need = np.cumprod(wl1_for,axis=0) * pc1 * pk1 * w_width 
                #* self.pk2 * self.pla2
            wl1_need[wl1_need > WL1] = mem_minimum
            wl1_cp[i] = wl1_need.shape[0] - 1 - np.argmin(np.flip(wl1_need,axis=0),axis=0)
            # 计算L2->L1,cp点(含)以内考虑相关性,其他不考虑
            wl2_to_wl1_tmp = np.ones(self.population_for)
            for j in wl1_cp.shape[1]:
                wl2_to_wl1_tmp[j] = wl1_need[wl1_cp[i][j]][j] * np.prod(pe_serial_for_reorder[i][wl1_cp[i][j]+1:,j])
            wl2_to_wl1[i] = wl2_to_wl1_tmp * np.prod(noc_serial_for,axis=0) * np.prod(nop_serial_for,axis=0)
            ## OL1
            ol1_for = pe_serial_for * (np.expand_dims(O_correlation,axis=0).repeat(self.population_for,axis=0).T)
            ol1_for[ol1_for <= 0] = 1 
            ol1_for = ol1_for[self.pe_order[i]] 
            ol1_need = np.cumprod(ol1_for,axis=0) * pk1 * o_width 
                #* self.pp2 * self.pq2 * self.pk2 * self.pla2
            ol1_need[ol1_need > OL1] = mem_minimum
            ol1_cp[i] = ol1_need.shape[0] - 1 - np.argmax(np.flip(ol1_need,axis=0),axis=0)
            # 计算L2->L1,cp点(含)以内考虑相关性,其他不考虑
            ol2_to_ol1_tmp = np.ones(self.population_for)
            for j in ol1_cp.shape[1]:
                ol2_to_ol1_tmp[j] = ol1_need[ol1_cp[i][j]][j] * np.prod(pe_serial_for_reorder[i][ol1_cp[i][j]+1:,j])
            ol2_to_ol1[i] = ol2_to_ol1_tmp * np.prod(noc_serial_for,axis=0) * np.prod(nop_serial_for,axis=0)
            ## DDR->L2
            ## AL2
            # 考虑相关性
            al2_for = l2_serial_for * (np.expand_dims(l2_A_correlation,axis=0).repeat(self.population_for,axis=0).T)
            al2_for[al2_for <= 0] = 1 
            # 按照顺序重新排列
            al2_for = al2_for[l2_order[i]] 
            l2_serial_for_reorder[i] = np.vstack([l2_serial_for[l2_order[i]],np.ones(self.population_for)])
            # 找到cp点
            al2_need = np.cumprod(al2_for,axis=0) * pc1 * a_width \
                * self.pp2 * self.pq2 * self.pla2 #* self.pp3 * self.pq3 * self.pla3 
            al2_need[al2_need > AL2] = mem_minimum
            al2_cp[i] = al2_need.shape[0] - 1 - np.argmax(np.flip(al2_need,axis=0),axis=0)
            # 计算ddr->L2,cp点(含)以内考虑相关性,其他不考虑
            ddr_to_al2_tmp = np.ones(self.population_for)
            for j in al2_cp.shape[1]:
                ddr_to_al2_tmp[j] = al2_need[al2_cp[i][j]][j] * np.prod(l2_serial_for_reorder[i][al2_cp[i][j]+1:,j])
            ddr_to_al2[i] = ddr_to_al2_tmp * np.prod(nop_serial_for,axis=0)
            ## WL2
            wl2_for = l2_serial_for * (np.expand_dims(l2_W_correlation,axis=0).repeat(self.population_for,axis=0).T)
            wl2_for[wl2_for <= 0] = 1 
            wl2_for = wl2_for[l2_order[i]] 
            wl2_need = np.cumprod(wl2_for,axis=0) * pc1 * pk1 * w_width \
                * self.pk2 * self.pla2 #* self.pk3 * self.pla3
            wl2_need[wl2_need > WL2] = mem_minimum
            wl2_cp[i] = wl2_need.shape[0] - 1 - np.argmax(wl2_need,axis=0)
            # 计算ddr->L2,cp点(含)以内考虑相关性,其他不考虑
            ddr_to_wl2_tmp = np.ones(self.population_for)
            for j in wl2_cp.shape[1]:
                ddr_to_wl2_tmp[j] = wl2_need[wl2_cp[i][j]][j] * np.prod(l2_serial_for_reorder[i][wl2_cp[i][j]+1:,j])
            ddr_to_wl2[i] = ddr_to_wl2_tmp * np.prod(nop_serial_for,axis=0)
            ## OL2
            ol2_for = l2_serial_for * (np.expand_dims(l2_O_correlation,axis=0).repeat(self.population_for,axis=0).T)
            ol2_for[ol2_for <= 0] = 1 
            ol2_for = ol2_for[l2_order[i]] 
            ol2_need = np.cumprod(ol2_for,axis=0) * pk1 * o_width \
                * self.pp2 * self.pq2 * self.pk2 * self.pla2 #* self.pp3 * self.pq3 * self.pk3 * self.pla3
            ol2_need[ol2_need > OL2] = mem_minimum
            ol2_cp[i] = ol2_need.shape[0] - np.argmax(ol2_need,axis=0)
            # 计算ddr->L2,cp点(不含)以内考虑相关性,其他不考虑
            ddr_to_ol2_tmp = np.ones(self.population_for)
            for j in ol2_cp.shape[1]:
                ddr_to_ol2_tmp[j] = ol2_need[ol2_cp[i][j]][j] * np.prod(l2_serial_for_reorder[i][ol2_cp[i][j]+1:,j])
            ddr_to_ol2[i] = ddr_to_ol2_tmp * np.prod(nop_serial_for,axis=0)
            
            # 只仿真L2,找到最小循环
            l2_inner_cp = np.minimum(al2_cp[i],wl2_cp[i],ol2_cp[i])
            # cp点与最小循环的循环次数之比
            for j in len(acp_core_ratio):
                acp_core_ratio[j] = np.prod(l2_serial_for_reorder[i][l2_inner_cp[j]+1:al2_cp[i][j]+1,j])
                wcp_core_ratio[j] = np.prod(l2_serial_for_reorder[i][l2_inner_cp[j]+1:wl2_cp[i][j]+1,j])
                ocp_core_ratio[j] = np.prod(l2_serial_for_reorder[i][l2_inner_cp[j]+1:ol2_cp[i][j]+1,j])
            # 最小循环数据传输量
            core_a_data[i] = al2_need[al2_cp[i]][:] / acp_core_ratio
            core_w_data[i] = wl2_need[wl2_cp[i]][:] / wcp_core_ratio
            core_o_data[i] = ol2_need[ol2_cp[i]][:] / ocp_core_ratio
            # 最小循环计算cycle数
            core_computation[i] = np.prod(l2_serial_for_reorder[i][:l2_inner_cp+1],axis=0)


