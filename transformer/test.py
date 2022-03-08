from GA import *

network_param = {"P":56,"Q":56,"R":1,"S":1,"C":256,"K":128,"LA":1,"stride":1}


if __name__ == "__main__":
	# --- 硬件参数
	HW_param = {"Chiplet":[4,4],"PE":[4,4],"intra_PE":{"C":8,"K":8}}       	
	memory_param = {"OL1":3 ,"OL2":48,"AL1":8,"AL2":48,"WL1":32,"WL2":32} #Byte
	
	NoC_w = HW_param["PE"][1] + 1
	NOC_NODE_NUM = NoC_w * HW_param["PE"][0]
	NoP_w = HW_param["Chiplet"][1] + 1
	NOP_SIZE = NoP_w * HW_param["Chiplet"][0]
	TOPO_param = {"NoC_w":NoC_w, "NOC_NODE_NUM": NOC_NODE_NUM, "NoP_w": NoP_w, "NOP_SIZE": NOP_SIZE,"nop_scale_ratio": nop_bandwidth/noc_bandwidth}

	example = GA(network_param,HW_param,debug=1)
	example.evaluate()
