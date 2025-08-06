##################################################
#                 Configurations                 #
##################################################

#
# Datasets
#

# label type
LABEL_IMAGE = ++trainer.label_type="image"
LABEL_HTML = ++trainer.label_type="html" "++trainer.train.loss_weights.html=1"
LABEL_CELL = ++trainer.label_type="cell" "++trainer.train.loss_weights.cell=1"
LABEL_BBOX = ++trainer.label_type="bbox" "++trainer.train.loss_weights.bbox=1"

LABEL_MIX = ++trainer.label_type="mix"\
			"++trainer.train.loss_weights.mix_loss=1"\
			"++trainer.train.loss_weights.html=1"\
			"++trainer.train.loss_weights.bbox=1"\

LABEL_MIX_CP = ++trainer.label_type="mix_cp"\
			"++trainer.train.loss_weights.cp=1"\
			"++trainer.train.loss_weights.html=1"\
			"++trainer.train.loss_weights.bbox=1"\
			"++trainer.train.loss_weights.mix_loss=1"\




TRAINER_TABLE = trainer=table_mix

MEAN = [0.86597056,0.88463002,0.87491087]
STD = [0.20686628,0.18201602,0.18485524]

# multiple datasets
DATA_MULTI = dataset=concat_dataset

#
# Vocab
#
VOCAB_NONE = vocab=empty
VOCAB_HTML = vocab=html
VOCAB_BBOX = vocab=bbox
VOCAB_CELL = vocab=cell
VOCAB_MIX = vocab=mix
VOCAB_MIX_OTSL = vocab=mix_otsl



# Trainer all
I448 = ++trainer.img_size=[448,448]
MODEL_VQVAE = model=vqvae
MODEL_VQVAE_L = $(MODEL_VQVAE) ++model.codebook_tokens=16384 ++model.hidden_dim=512
VQVAE1M_WEIGHTS = $(MODEL_VQVAE) ++trainer.vqvae_weights="../experiments/unitable_weights/vqvae_1m.pt"
VQVAE2M_WEIGHTS = $(MODEL_VQVAE_L) ++trainer.vqvae_weights="../experiments/unitable_weights/vqvae_2m.pt"
WEIGHTS_mtim_2m_base = ++trainer.trainer.beit_pretrained_weights="../experiments/unitable_weights/ssp_2m_base.pt"
WEIGHTS_mtim_2m_large = ++trainer.trainer.beit_pretrained_weights="../experiments/unitable_weights/ssp_2m_large.pt"
WEIGHTS_mtim_2m_large_KO = ++trainer.trainer.beit_pretrained_weights="../experiments/unitable_weights/ssp_2m_large_ko.pt"
LOCK_MTIM_4 = ++trainer.trainer.freeze_beit_epoch=4




# lr + scheduler
MODEL_BEIT = model=beit
IMGLINEAR = model/model/backbone=imglinear
NHEAD12 = ++model.nhead=12
FF4 = ++model.ff_ratio=4
NORM_FIRST = ++model.norm_first=true
ACT_GELU = ++model.activation="gelu"
D_MODEL768 = ++model.d_model=768
REG_d02 = ++model.dropout=0.2
P16 = ++model.backbone_downsampling_factor=16
E12 = ++model.model.encoder.nlayer=12


# SSP Model Parameters
NHEAD8 = ++model.nhead=8
D_MODEL512 = ++model.d_model=512
E4 = ++model.model.encoder.nlayer=4
OTHRES = ++trainer.trans_size=[448,448] ++trainer.vqvae_size=[224,224] ++trainer.grid_size=28 ++trainer.num_mask_patches=300

MODEL_ENCODER_DECODER = model=encoderdecoder
MODEL_SHARED_DUAL = model=sharedencoder_dualdecoder
MODEL_HIERACHICAL = model=HierarchicalSharedEncoder

D4 = ++model.model.decoder_html.nlayer=4 ++model.model.decoder_bbox.nlayer=4

# LR_cosine = trainer/train/lr_scheduler=cosine ++trainer.train.lr_scheduler.lr_lambda.min_ratio=5e-3


# augmentation
AUG_RESIZE_NORM = dataset/augmentation=resize_normalize \
	++dataset.augmentation.transforms.2.mean=$(MEAN) ++dataset.augmentation.transforms.2.std=$(STD)


# [ Structure ]

# Cosine
LR_8e5 = ++trainer.train.optimizer_html.lr=8e-5 
LR_cosine_html = trainer/train/lr_scheduler_html=cosine ++trainer.train.lr_scheduler_html.lr_lambda.min_ratio=5e-3
LR_cosine93k_warm6k = $(LR_cosine_html) ++trainer.train.lr_scheduler_html.lr_lambda.total_step=93400 ++trainer.train.lr_scheduler_html.lr_lambda.warmup=5800 # Structure




SEQ512_html = trainer.max_seq_len_html=512
SEQ1024_html = trainer.max_seq_len_html=1024
OPT_ADAMW_HTML = trainer/train/optimizer_html=adamw
OPT_WD5e2_HTML = ++trainer.train.optimizer_html.weight_decay=5e-2


# [ BBOX ]
GRAD_CLIP12 = ++trainer.train.grad_clip_bbox=12
LR_3e4 = ++trainer.train.optimizer_bbox.lr=3e-4 
LR_cosine_bbox = trainer/train/lr_scheduler_bbox=cosine ++trainer.train.lr_scheduler_bbox.lr_lambda.min_ratio=5e-3
LR_cosine77k_warm8k = $(LR_cosine_bbox) ++trainer.train.lr_scheduler_bbox.lr_lambda.total_step=76600 ++trainer.train.lr_scheduler_bbox.lr_lambda.warmup=7660
# LR_cosine153k_warm15k = $(LR_cosine_bbox) ++trainer.train.lr_scheduler_bbox.lr_lambda.total_step=15320 ++trainer.train.lr_scheduler_bbox.lr_lambda.warmup=15320

SEQ1024_bbox = trainer.max_seq_len_bbox=1024 
SEQ2048_bbox = trainer.max_seq_len_bbox=2048 
OPT_ADAMW_BBOX = trainer/train/optimizer_bbox=adamw
OPT_WD5e2_BBOX = ++trainer.train.optimizer_bbox.weight_decay=5e-2




#######################
#		TESTING
#######################


# DATASET : Public
PUBTABNET_M = +dataset/pubtabnet@dataset.train.d1=train_dataset \
	+dataset/pubtabnet@dataset.valid.d1=valid_dataset \
	+dataset/pubtabnet@dataset.test.d1=test_dataset

SYN_FIN_M = +dataset/synthtabnet_fintabnet@dataset.train.d2=train_dataset \
	+dataset/synthtabnet_fintabnet@dataset.valid.d2=valid_dataset \
	+dataset/synthtabnet_fintabnet@dataset.test.d2=test_dataset
SYN_MRK_M = +dataset/synthtabnet_marketing@dataset.train.d3=train_dataset \
	+dataset/synthtabnet_marketing@dataset.valid.d3=valid_dataset \
	+dataset/synthtabnet_marketing@dataset.test.d3=test_dataset
SYN_PUB_M = +dataset/synthtabnet_pubtabnet@dataset.train.d4=train_dataset \
	+dataset/synthtabnet_pubtabnet@dataset.valid.d4=valid_dataset \
	+dataset/synthtabnet_pubtabnet@dataset.test.d4=test_dataset
SYN_SPA_M = +dataset/synthtabnet_sparse@dataset.train.d5=train_dataset \
	+dataset/synthtabnet_sparse@dataset.valid.d5=valid_dataset \
	+dataset/synthtabnet_sparse@dataset.test.d5=test_dataset

FINTABNET_M = +dataset/fintabnet_processed@dataset.train.d6=train_dataset \
	+dataset/fintabnet_processed@dataset.valid.d6=valid_dataset \
	+dataset/fintabnet_processed@dataset.test.d6=test_dataset

DATA_PTN = $(DATA_MULTI) \
	$(PUBTABNET_M)
DATA_FIN = $(DATA_MULTI) \
	$(FINTABNET_M)
DATA_PUB = $(DATA_MULTI) \
	$(PUBTABNET_M) $(SYN_FIN_M) $(SYN_MRK_M) $(SYN_SPA_M) $(SYN_PUB_M) $(FINTABNET_M)


# DATASET
MINIPUBTABNET_M = +dataset/mini_pubtabnet@dataset.train.d8=train_dataset \
	+dataset/mini_pubtabnet@dataset.valid.d8=valid_dataset \
	+dataset/mini_pubtabnet@dataset.test.d8=test_dataset

PTN_40000_M = +dataset/pubtabnet_40000@dataset.train.d1=train_dataset \
	+dataset/pubtabnet_40000@dataset.valid.d1=valid_dataset \
	+dataset/pubtabnet_40000@dataset.test.d1=test_dataset
FIN_20000_M = +dataset/fintabnet_processed_20000@dataset.train.d2=train_dataset \
	+dataset/fintabnet_processed_20000@dataset.valid.d2=valid_dataset \
	+dataset/fintabnet_processed_20000@dataset.test.d2=test_dataset
SYN_PUB_10000_M = +dataset/synthtabnet_pubtabnet_10000@dataset.train.d3=train_dataset \
	+dataset/synthtabnet_pubtabnet_10000@dataset.valid.d3=valid_dataset \
	+dataset/synthtabnet_pubtabnet_10000@dataset.test.d3=test_dataset
SYN_FIN_10000_M = +dataset/synthtabnet_fintabnet_10000@dataset.train.d4=train_dataset \
	+dataset/synthtabnet_fintabnet_10000@dataset.valid.d4=valid_dataset \
	+dataset/synthtabnet_fintabnet_10000@dataset.test.d4=test_dataset
SYN_MKT_10000_M = +dataset/synthtabnet_marketing_10000@dataset.train.d5=train_dataset \
	+dataset/synthtabnet_marketing_10000@dataset.valid.d5=valid_dataset \
	+dataset/synthtabnet_marketing_10000@dataset.test.d5=test_dataset
SYN_SPS_10000_M = +dataset/synthtabnet_sparse_10000@dataset.train.d6=train_dataset \
	+dataset/synthtabnet_sparse_10000@dataset.valid.d6=valid_dataset \
	+dataset/synthtabnet_sparse_10000@dataset.test.d6=test_dataset

PTN_100K_M = +dataset/pubtabnet_100K@dataset.train.d1=train_dataset \
	+dataset/pubtabnet_100K@dataset.valid.d1=valid_dataset \
	+dataset/pubtabnet_100K@dataset.test.d1=test_dataset

PTN_50K_M = +dataset/pubtabnet_50K@dataset.train.d1=train_dataset \
	+dataset/pubtabnet_50K@dataset.valid.d1=valid_dataset \
	+dataset/pubtabnet_50K@dataset.test.d1=test_dataset
SYN_PUB_12K_M = +dataset/synthtabnet_pubtabnet_12K@dataset.train.d3=train_dataset \
	+dataset/synthtabnet_pubtabnet_12K@dataset.valid.d3=valid_dataset \
	+dataset/synthtabnet_pubtabnet_12K@dataset.test.d3=test_dataset
SYN_FIN_12K_M = +dataset/synthtabnet_fintabnet_12K@dataset.train.d4=train_dataset \
	+dataset/synthtabnet_fintabnet_12K@dataset.valid.d4=valid_dataset \
	+dataset/synthtabnet_fintabnet_12K@dataset.test.d4=test_dataset
SYN_MKT_12K_M = +dataset/synthtabnet_marketing_12K@dataset.train.d5=train_dataset \
	+dataset/synthtabnet_marketing_12K@dataset.valid.d5=valid_dataset \
	+dataset/synthtabnet_marketing_12K@dataset.test.d5=test_dataset
SYN_SPS_12K_M = +dataset/synthtabnet_sparse_12K@dataset.train.d6=train_dataset \
	+dataset/synthtabnet_sparse_12K@dataset.valid.d6=valid_dataset \
	+dataset/synthtabnet_sparse_12K@dataset.test.d6=test_dataset


PTN_505K_M = +dataset/pubtabnet_505K@dataset.train.d1=train_dataset \
	+dataset/pubtabnet_505K@dataset.valid.d1=valid_dataset \
	+dataset/pubtabnet_505K@dataset.test.d1=test_dataset
FIN_98K_M = +dataset/fintabnet_processed_98K@dataset.train.d2=train_dataset \
	+dataset/fintabnet_processed_98K@dataset.valid.d2=valid_dataset \
	+dataset/fintabnet_processed_98K@dataset.test.d2=test_dataset
SYN_PUB_140K_M = +dataset/synthtabnet_pubtabnet_140K@dataset.train.d3=train_dataset \
	+dataset/synthtabnet_pubtabnet_140K@dataset.valid.d3=valid_dataset \
	+dataset/synthtabnet_pubtabnet_140K@dataset.test.d3=test_dataset
SYN_FIN_140K_M = +dataset/synthtabnet_fintabnet_140K@dataset.train.d4=train_dataset \
	+dataset/synthtabnet_fintabnet_140K@dataset.valid.d4=valid_dataset \
	+dataset/synthtabnet_fintabnet_140K@dataset.test.d4=test_dataset
SYN_MKT_140K_M = +dataset/synthtabnet_marketing_140K@dataset.train.d5=train_dataset \
	+dataset/synthtabnet_marketing_140K@dataset.valid.d5=valid_dataset \
	+dataset/synthtabnet_marketing_140K@dataset.test.d5=test_dataset
SYN_SPS_150K_M = +dataset/synthtabnet_sparse_150K@dataset.train.d6=train_dataset \
	+dataset/synthtabnet_sparse_150K@dataset.valid.d6=valid_dataset \
	+dataset/synthtabnet_sparse_150K@dataset.test.d6=test_dataset

################## GENERATOR ##################

GEN_IMAGE_1000 = +dataset/generator/gen_2025_03_21_image_1000@dataset.train.d7=train_dataset \
	+dataset/generator/gen_2025_03_21_image_1000@dataset.valid.d7=valid_dataset \
	+dataset/generator/gen_2025_03_21_image_1000@dataset.test.d7=test_dataset

GEN_FORMULA_1000 = +dataset/generator/gen_2025_03_21_formula_1000@dataset.train.d8=train_dataset \
	+dataset/generator/gen_2025_03_21_formula_1000@dataset.valid.d8=valid_dataset \
	+dataset/generator/gen_2025_03_21_formula_1000@dataset.test.d8=test_dataset




################## IX Dataset ##################
IX_mvp27_case1_250411 = +dataset/ix_dataset/MVP3_target_case1_250411@dataset.train.d11=train_dataset \
	+dataset/ix_dataset/MVP3_target_case1_250411@dataset.valid.d11=valid_dataset \
	+dataset/ix_dataset/MVP3_target_case1_250411@dataset.test.d11=test_dataset

IX_mvp27_case3_250411 = +dataset/ix_dataset/MVP3_target_case3_250411@dataset.train.d13=train_dataset \
	+dataset/ix_dataset/MVP3_target_case3_250411@dataset.valid.d13=valid_dataset \
	+dataset/ix_dataset/MVP3_target_case3_250411@dataset.test.d13=test_dataset

IX_mvp27_case4_250411 = +dataset/ix_dataset/MVP3_target_case4_250411@dataset.train.d14=train_dataset \
	+dataset/ix_dataset/MVP3_target_case4_250411@dataset.valid.d14=valid_dataset \
	+dataset/ix_dataset/MVP3_target_case4_250411@dataset.test.d14=test_dataset

IX_mvp27_case_old_250325 = +dataset/ix_dataset/MVP3_target_case_old_250325@dataset.train.d15=train_dataset \
	+dataset/ix_dataset/MVP3_target_case_old_250325@dataset.valid.d15=valid_dataset \
	+dataset/ix_dataset/MVP3_target_case_old_250325@dataset.test.d15=test_dataset

# 300장
IX_mvp30_case1_250509 = +dataset/ix_dataset/MVP30_target_case1_250509_300@dataset.train.d16=train_dataset \
	+dataset/ix_dataset/MVP30_target_case1_250509_300@dataset.valid.d16=valid_dataset \
	+dataset/ix_dataset/MVP30_target_case1_250509_300@dataset.test.d16=test_dataset

# 300장
IX_mvp30_case2_250617 = +dataset/ix_dataset/MVP30_target_case2_250617_1_300@dataset.train.d17=train_dataset \
	+dataset/ix_dataset/MVP30_target_case2_250617_1_300@dataset.valid.d17=valid_dataset \
	+dataset/ix_dataset/MVP30_target_case2_250617_1_300@dataset.test.d17=test_dataset


IX_mvp4_case3_injection_molding = +dataset/ix_dataset/MVP4_target_case3_injection_molding@dataset.train.d20=train_dataset \
	+dataset/ix_dataset/MVP4_target_case3_injection_molding@dataset.valid.d20=valid_dataset \
	+dataset/ix_dataset/MVP4_target_case3_injection_molding@dataset.test.d20=test_dataset


DATA_PTN_100K = $(DATA_MULTI) \
	$(PTN_100K_M)

DATA_1M_NO_FIN = $(DATA_MULTI) \
	$(PTN_505K_M) $(SYN_PUB_140K_M) $(SYN_FIN_140K_M) $(SYN_MKT_140K_M) $(SYN_SPS_150K_M)

DATA_SYN = $(DATA_MULTI) \
	$(SYN_PUB_140K_M) $(SYN_FIN_140K_M) $(SYN_MKT_140K_M) $(SYN_SPS_150K_M)

DATA_MINI = $(DATA_MULTI) \
	$(MINIPUBTABNET_M)

DATA_AB_CLEAN = $(DATA_MULTI) \
	$(PTN_40000_M) $(FIN_20000_M) $(SYN_PUB_10000_M) $(SYN_FIN_10000_M) $(SYN_MKT_10000_M) $(SYN_SPS_10000_M) 

DATA_AB_CLEAN_500K = $(DATA_MULTI) \
	$(PTN_240000_M) $(FIN_60000_M) $(SYN_PUB_50000_M) $(SYN_FIN_50000_M) $(SYN_MKT_50000_M) $(SYN_SPS_50000_M) 

DATA_FT_100K = $(DATA_MULTI) \
	$(PTN_50K_M) $(SYN_PUB_12K_M) $(SYN_FIN_12K_M) $(SYN_MKT_12K_M) $(SYN_SPS_12K_M)\
	$(GEN_IMAGE_1000) $(GEN_FORMULA_1000) $(IX_mvp27_case1_250411) $(IX_mvp27_case3_250411) $(IX_mvp27_case4_250411) 


DATA_1M_IX = $(DATA_MULTI) \
	$(PTN_505K_M) $(SYN_PUB_140K_M) $(SYN_FIN_140K_M) $(SYN_MKT_140K_M) $(SYN_SPS_150K_M)\
	$(GEN_IMAGE_1000) $(GEN_FORMULA_1000) \
	$(IX_mvp27_case1_250411) $(IX_mvp27_case3_250411) $(IX_mvp27_case4_250411) $(IX_mvp27_case_old_250325) 


DATA_1M_IX_MVP30 = $(DATA_MULTI) \
	$(PTN_505K_M) $(SYN_PUB_140K_M) $(SYN_FIN_140K_M) $(SYN_MKT_140K_M) $(SYN_SPS_150K_M)\
	$(GEN_IMAGE_1000) $(GEN_FORMULA_1000) \
	$(IX_mvp27_case1_250411) $(IX_mvp27_case3_250411) $(IX_mvp27_case4_250411) $(IX_mvp27_case_old_250325) \
	$(IX_mvp30_case1_250509) $(IX_mvp30_case2_250617)


DATA_1M_IX_MVP40 = $(DATA_MULTI) \
	$(PTN_505K_M) $(SYN_PUB_140K_M) $(SYN_FIN_140K_M) $(SYN_MKT_140K_M) $(SYN_SPS_150K_M)\
	$(GEN_IMAGE_1000) $(GEN_FORMULA_1000) \
	$(IX_mvp27_case1_250411) $(IX_mvp27_case3_250411) $(IX_mvp27_case4_250411) $(IX_mvp27_case_old_250325) \
	$(IX_mvp30_case1_250509) $(IX_mvp30_case2_250617) \
	$(IX_mvp4_case3_injection_molding)





DATA_FT_IX = $(DATA_MULTI) $(IX_mvp27_case1_250411) $(IX_mvp27_case3_250411) $(IX_mvp27_case4_250411)
DATA_FT_IX_TRAIN = $(DATA_MULTI) $(IX_mvp27_case1_250411_train) $(IX_mvp27_case3_250411_train) $(IX_mvp27_case4_250411_train)



BATCH1 = ++trainer.train.dataloader.batch_size=1 ++trainer.valid.dataloader.batch_size=1
BATCH8 = ++trainer.train.dataloader.batch_size=8 ++trainer.valid.dataloader.batch_size=8
BATCH16 = ++trainer.train.dataloader.batch_size=16 ++trainer.valid.dataloader.batch_size=16
BATCH24 = ++trainer.train.dataloader.batch_size=24 ++trainer.valid.dataloader.batch_size=24
BATCH32 = ++trainer.train.dataloader.batch_size=32 ++trainer.valid.dataloader.batch_size=32
BATCH48 = ++trainer.train.dataloader.batch_size=48 ++trainer.valid.dataloader.batch_size=48
BATCH64 = ++trainer.train.dataloader.batch_size=64 ++trainer.valid.dataloader.batch_size=64
BATCH72 = ++trainer.train.dataloader.batch_size=72 ++trainer.valid.dataloader.batch_size=72
BATCH96 = ++trainer.train.dataloader.batch_size=96 ++trainer.valid.dataloader.batch_size=96
BATCH128 = ++trainer.train.dataloader.batch_size=128 ++trainer.valid.dataloader.batch_size=128
BATCH256 = ++trainer.train.dataloader.batch_size=256 ++trainer.valid.dataloader.batch_size=256
BATCH384 = ++trainer.train.dataloader.batch_size=384 ++trainer.valid.dataloader.batch_size=384

EPOCH3 = ++trainer.train.epochs=3
EPOCH5 = ++trainer.train.epochs=5
EPOCH7 = ++trainer.train.epochs=7
EPOCH10 = ++trainer.train.epochs=10
EPOCH20 = ++trainer.train.epochs=20
EPOCH24 = ++trainer.train.epochs=24
EPOCH31 = ++trainer.train.epochs=31
EPOCH48 = ++trainer.train.epochs=48
EPOCH25 = ++trainer.train.epochs=25
EPOCH30 = ++trainer.train.epochs=30
EPOCH32 = ++trainer.train.epochs=32
EPOCH40 = ++trainer.train.epochs=40
EPOCH50 = ++trainer.train.epochs=50
EPOCH60 = ++trainer.train.epochs=60
EPOCH100 = ++trainer.train.epochs=70


# IX_200
# DATA = $(DATA_MULTI)\
# 		$(GEN_IMAGE_2000) $(GEN_FORMULA_1500) $(GEN_FORMULA_500)


# $(GEN_IMAGE_1000)
# DATA_MINI / DATA_FIN / DATA_PTN
# GEN_FORMULA_1000 / GEN_IMAGE_1000
USE_HTML += ++trainer.trainer.otsl_mode=False
USE_AUG = ++trainer.train.dataloader.aug=True

EPOCH = $(EPOCH48)
BATCH = $(BATCH72)


MTIM_BASE = $(MODEL_BEIT) $(IMGLINEAR) $(NHEAD8) $(FF4) $(ACT_GELU) \
	$(NORM_FIRST) $(D_MODEL512) $(REG_d02) $(P16) $(E4) $(OTHRES)
MTIM_LARGE = $(MODEL_BEIT) $(IMGLINEAR) $(NHEAD12) $(FF4) $(ACT_GELU) \
	$(NORM_FIRST) $(D_MODEL768) $(REG_d02) $(P16) $(E12)

ARCH_BASE = $(MTIM_BASE) $(MODEL_ENCODER_DECODER) $(D4)
ARCH_LARGE = $(MTIM_LARGE) $(MODEL_ENCODER_DECODER) $(D4)



COMMONS_BASE := $(VOCAB_MIX) $(USE_HTML)\
			$(WEIGHTS_mtim_2m_base) $(ARCH_BASE)\
			$(EPOCH)\
			$(BATCH)\
			$(LABEL_MIX) $(AUG_RESIZE_NORM)\
			$(TRAINER_TABLE) $(I448) $(OPT_ADAMW)\
			$(LOCK_MTIM_4)
			

COMMONS_LARGE := $(VOCAB_MIX) $(USE_HTML)\
			$(WEIGHTS_mtim_2m_large) $(ARCH_LARGE)\
			$(EPOCH)\
			$(BATCH)\
			$(LABEL_MIX) $(AUG_RESIZE_NORM)\
			$(TRAINER_TABLE) $(I448) $(OPT_ADAMW)\
			$(LOCK_MTIM_4)



# SharedEncoder_DualDecoder uses unified optimizer and scheduler
SHARED_OPTIMIZER := ++trainer.train.optimizer.lr=1e-4 \
		++trainer.train.optimizer._target_=torch.optim.AdamW \
		++trainer.train.optimizer.weight_decay=5e-2 \
		++trainer.train.grad_clip=12

# Legacy HTML/BBOX settings (for backward compatibility) 
HTML := $(SEQ512_html)  $(OPT_ADAMW_HTML) $(OPT_WD5e2_HTML) $(LR_8e5) $(LR_cosine93k_warm6k)
BBOX := $(SEQ1024_bbox) $(OPT_ADAMW_BBOX) $(OPT_WD5e2_BBOX) $(LR_3e4) $(LR_cosine77k_warm8k)\
		$(GRAD_CLIP12)


USE_OTSL += ++trainer.trainer.otsl_mode=True
USE_MIX_LOSS += ++trainer.trainer.use_mix_loss=True
FT_MODE = ++trainer.trainer.finetune_mode=True



OTSL_MODE = $(USE_OTSL) $(VOCAB_MIX_OTSL)






############## MVP4.0 : Learning Rate 테스트(Max Token 유지) #######################
# HTML : 512 > 1024
# BBOX : 1024 > 2048
# Batch : 16
# Batch Accumulation : HTML 6 / BBOX 4


# HTML
LR_2e4_html  = ++trainer.train.optimizer_html.lr=2e-4     # √batch 스케일 후 값
LR_cosine125k_warm3k = \
  trainer/train/lr_scheduler_html=cosine \
  ++trainer.train.lr_scheduler_html.lr_lambda.total_step=124992 \
  ++trainer.train.lr_scheduler_html.lr_lambda.warmup=3750 \
  ++trainer.train.lr_scheduler_html.lr_lambda.min_ratio=5e-3 \
  ++trainer.train.lr_scheduler_html.lr_lambda.cycle=1.5

# BBOX
LR_2e4_bbox  = ++trainer.train.optimizer_bbox.lr=2e-4
LR_cosine187k_warm5k = \
  trainer/train/lr_scheduler_bbox=cosine \
  ++trainer.train.lr_scheduler_bbox.lr_lambda.total_step=187488 \
  ++trainer.train.lr_scheduler_bbox.lr_lambda.warmup=5625 \
  ++trainer.train.lr_scheduler_bbox.lr_lambda.min_ratio=5e-3 \
  ++trainer.train.lr_scheduler_bbox.lr_lambda.cycle=1.5

lr_setting = $(LR_2e4_html) $(LR_cosine125k_warm3k) $(LR_2e4_bbox) $(LR_cosine187k_warm5k)

# make experiments/mvp4_lr_cycle_15/.done_finetune
EXP_mvp4_lr_cycle_15 := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA_1M_IX_MVP30)\
					$(BATCH16) $(EPOCH48) $(OTSL_MODE) $(USE_MIX_LOSS) $(USE_AUG)\
					$(IX_mvp4_case3_injection_molding)\
					$(SEQ1024_html) $(SEQ2048_bbox)\
					$(lr_setting)




############## SharedEncoder_DualDecoder Experiments #######################

# SharedEncoder_DualDecoder specific settings (updated with sequence lengths)
SHARED_OPTIMIZER_EXTENDED := ++trainer.train.optimizer.lr=2e-4 \
		++trainer.train.optimizer._target_=torch.optim.AdamW \
		++trainer.train.optimizer.weight_decay=5e-2 \
		++trainer.train.grad_clip=12

LR_seq = trainer/train/lr_scheduler=sequential

SETTING := $(LR_seq) $(SHARED_OPTIMIZER_EXTENDED) $(FT_MODE) $(USE_AUG) $(OTSL_MODE) $(USE_MIX_LOSS)
 
# make experiments/mvp4_lr_sequential_try2_SYN_shared/.done_finetune
EXP_mvp4_lr_sequential_try2_SYN_shared := $(COMMONS_LARGE) $(SETTING) $(DATA_SYN)\
					$(BATCH16) $(EPOCH48)\
					$(SEQ1024_html) $(SEQ2048_bbox)



# make experiments/mvp4_lr_sequential_try2_SYN_shared_D4_6/.done_finetune
EXP_mvp4_lr_sequential_try2_SYN_shared_D4_6 := $(COMMONS_LARGE) $(SETTING) $(DATA_SYN)\
					$(BATCH16) $(EPOCH48)\
					$(SEQ1024_html) $(SEQ2048_bbox) $(D4_6)







D4_4 = ++model.model.decoder_html.nlayer=4 ++model.model.decoder_bbox.nlayer=4
D8_8 = ++model.model.decoder_html.nlayer=8 ++model.model.decoder_bbox.nlayer=8
D12_12 = ++model.model.decoder_html.nlayer=12 ++model.model.decoder_bbox.nlayer=12


######### BASE #########
# make experiments/mvp4_seq_lr_shared_D4_4_PTN_100K_e32/.done_finetune
EXP_mvp4_seq_lr_shared_D4_4_PTN_100K_e32 := $(COMMONS_LARGE) $(SETTING) $(DATA_PTN_100K)\
					$(BATCH8) $(EPOCH32)\
					$(SEQ512_html) $(SEQ2048_bbox) $(D4_4) $(MODEL_SHARED_DUAL)
# make experiments/mvp4_seq_lr_shared_D8_8_PTN_100K_e32/.done_finetune
EXP_mvp4_seq_lr_shared_D8_8_PTN_100K_e32 := $(COMMONS_LARGE) $(SETTING) $(DATA_PTN_100K)\
					$(BATCH8) $(EPOCH32)\
					$(SEQ512_html) $(SEQ2048_bbox) $(D8_8) $(MODEL_SHARED_DUAL)





######### POS EMBED 분리 #########
# make experiments/mvp4_seq_lr_shared_D4_4_PTN_100K_e32_seperate_pos_emb/.done_finetune
EXP_mvp4_seq_lr_shared_D4_4_PTN_100K_e32_seperate_pos_emb := $(COMMONS_LARGE) $(SETTING) $(DATA_PTN_100K)\
					$(BATCH8) $(EPOCH32)\
					$(SEQ512_html) $(SEQ2048_bbox) $(D4_4) $(MODEL_SHARED_DUAL)



# make experiments/mvp4_seq_lr_shared_D4_4_PTN_100K_e32_seperate_pos_emb/.done_finetune
EXP_mvp4_seq_lr_shared_D4_4_PTN_100K_e48_seperate_pos_emb := $(COMMONS_LARGE) $(SETTING) $(DATA_PTN_100K)\
					$(BATCH8) $(EPOCH32)\
					$(SEQ1024_html) $(SEQ2048_bbox) $(D4_4) $(MODEL_SHARED_DUAL)










# # make experiments/mvp4_seq_lr_shared_D8_8_PTN_100K_e32_seperate_pos_emb/.done_finetune
# EXP_mvp4_seq_lr_shared_D8_8_PTN_100K_e32_seperate_pos_emb := $(COMMONS_LARGE) $(SETTING) $(DATA_PTN_100K)\
# 					$(BATCH8) $(EPOCH32)\
# 					$(SEQ512_html) $(SEQ2048_bbox) $(D8_8) $(MODEL_SHARED_DUAL)



######### Center Point Decoder 추가 #########

D4_4_4 = ++model.model.decoder_center.nlayer=4 ++model.model.decoder_html.nlayer=4 ++model.model.decoder_bbox.nlayer=4
D4_8_8 = ++model.model.decoder_center.nlayer=4 ++model.model.decoder_html.nlayer=8 ++model.model.decoder_bbox.nlayer=8

# make experiments/cp_seq_lr_shared_D4_4_4_PTN_100K_e32_seperate_pos_emb/.done_finetune
EXP_cp_seq_lr_shared_D4_4_4_PTN_100K_e32_seperate_pos_emb := $(COMMONS_LARGE) $(SETTING) $(DATA_PTN_100K)\
					$(BATCH8) $(EPOCH32)\
					$(SEQ512_html) $(SEQ2048_bbox) $(D4_4_4) $(MODEL_HIERACHICAL) $(LABEL_MIX_CP)



# make experiments/cp_seq_lr_shared_D4_8_8_PTN_100K_e32_seperate_pos_emb/.done_finetune
EXP_cp_seq_lr_shared_D4_8_8_PTN_100K_e32_seperate_pos_emb := $(COMMONS_LARGE) $(SETTING) $(DATA_PTN_100K)\
					$(BATCH8) $(EPOCH32)\
					$(SEQ512_html) $(SEQ2048_bbox) $(D4_8_8) $(MODEL_HIERACHICAL) $(LABEL_MIX_CP)



LOSS_WEIGHTS_CENTER_2 = "++trainer.train.loss_weights.cp=2"
# make experiments/cp_seq_lr_shared_D4_4_4_PTN_100K_e32_seperate_pos_emb_center_lw2/.done_finetune
EXP_cp_seq_lr_shared_D4_4_4_PTN_100K_e32_seperate_pos_emb_center_lw2 := $(COMMONS_LARGE) $(SETTING) $(DATA_PTN_100K)\
					$(BATCH8) $(EPOCH32)\
					$(SEQ512_html) $(SEQ2048_bbox) $(D4_8_8) $(MODEL_HIERACHICAL) $(LABEL_MIX_CP) $(LOSS_WEIGHTS_CENTER_2)





# make -C unitable_shared_encoder experiments/mvp4_seq_lr_shared_D4_4_PTN_100K_e32_seperate_pos_emb/.done_finetune && make -C unitable_shared_encoder experiments/mvp4_seq_lr_shared_D8_8_PTN_100K_e32_seperate_pos_emb/.done_finetune && make -C unitable_shared_encoder experiments/cp_seq_lr_shared_D4_4_4_PTN_100K_e32_seperate_pos_emb_center_lw2/.done_finetune && make -C unitable_shared_encoder experiments/cp_seq_lr_shared_D4_8_8_PTN_100K_e32_seperate_pos_emb/.done_finetune


# make -C unitable_shared_encoder experiments/mvp4_seq_lr_shared_D12_12_PTN_100K_e32/.done_finetune



# make experiments/mvp4_lr_sequential_try2_SYN_shared/.done_finetune & make experiments/mvp4_lr_sequential_try2_SYN_shared_D4_6/.done_finetune



# DATA_1M_IX_MVP30
# DATA_MINI



# D4_6




.PHONY: finetune_all
finetune_all:\
experiments/mvp30_adding_korean_table_in_ssp_E24/.done_finetune\



# make finetune_all