
from .emu3 import Emu3VisionVQModel, Emu3VisionVQImageProcessor

#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
 
def Emu3_VQ():
    return Emu3VisionVQModel 

VQ_models = {'Emu3_VQ': Emu3_VQ()}
