from .registry import LOSS_REGISTRY, register_loss, build_loss
from .cross_entropy import CrossEntropyLoss
from .triplet import TripletLoss
from .class_loss import ClassCELoss
from .circle_loss import CircleLoss
from .center_loss import CenterLoss
from .arcface_loss import ArcFaceLoss
from .subcenter_arcface_loss import SubCenterArcFaceLoss
from .ranked_list import RankedListLoss
from .combiner import ObjectiveCombiner
