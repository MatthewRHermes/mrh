import numpy as np

#bond lenghts all assumed to be constant
CsH=1.09
CsC=1.49
CdC=1.35
#bond angle is assumed is 120


origin=[0.0,0.0,0.0]
def generator(nfrags,origin=origin,CsH=CsH,CsC=CsC,CdC=CdC):
        atom_list=np.empty((nfrags*4+2,1),dtype='object')
        geom=np.empty((nfrags*4+2,3))
        geom[0]=[-CsH*np.cos(np.pi/6) ,CsH/2,0.0]
        atom_list[0]='H'
        for frag in range(nfrags):
                #print(origin)
                geom[1+4*frag]= origin
                atom_list[1+4*frag]='C'
                geom[2+4*frag]= geom[1+4*frag]+[CdC*np.cos(np.pi/6),CdC/2,0.0]
                atom_list[2+4*frag]='C'
                geom[3+4*frag]= geom[1+4*frag]-[0,CsH,0]
                atom_list[3+4*frag]='H'
                geom[4+4*frag]= geom[2+4*frag]+[0,CsH,0]
                atom_list[4+4*frag]='H'
                origin=geom[2+4*frag]+[CsC*np.cos(np.pi/6),-CsC/2,0.0]
        atom_list[-1]='H'
        geom[-1]=geom[-4]+[CsH*np.cos(np.pi/6),-CsH/2,0.0]
        #return np.hstack((atom_list,geom))
        return np.array2string(np.hstack((atom_list,geom))).replace('[','').replace(']',';').replace('\'','')
