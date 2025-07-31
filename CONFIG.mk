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
LABEL_MIX = ++trainer.label_type="mix" "++trainer.train.loss_weights.mix_combine=1" "++trainer.train.loss_weights.html=1" "++trainer.train.loss_weights.bbox=1"

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
D4 = ++model.model.decoder.nlayer=4

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
DATA_SYN = $(DATA_MULTI) \
	$(SYN_FIN_M) $(SYN_MRK_M) $(SYN_SPA_M)

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


# IX_mvp4_case1_200 = +dataset/ix_dataset/MVP4_target_case1_250623@dataset.train.d18=train_dataset \
# 	+dataset/ix_dataset/MVP4_target_case1_250623@dataset.valid.d18=valid_dataset \
# 	+dataset/ix_dataset/MVP4_target_case1_250623@dataset.test.d18=test_dataset

# IX_mvp4_case2_200 = +dataset/ix_dataset/MVP4_target_case2_250623@dataset.train.d19=train_dataset \
# 	+dataset/ix_dataset/MVP4_target_case2_250623@dataset.valid.d19=valid_dataset \
# 	+dataset/ix_dataset/MVP4_target_case2_250623@dataset.test.d19=test_dataset

IX_mvp4_case3_injection_molding = +dataset/ix_dataset/MVP4_target_case3_injection_molding@dataset.train.d20=train_dataset \
	+dataset/ix_dataset/MVP4_target_case3_injection_molding@dataset.valid.d20=valid_dataset \
	+dataset/ix_dataset/MVP4_target_case3_injection_molding@dataset.test.d20=test_dataset


DATA_1M_NO_FIN = $(DATA_MULTI) \
	$(PTN_505K_M) $(SYN_PUB_140K_M) $(SYN_FIN_140K_M) $(SYN_MKT_140K_M) $(SYN_SPS_150K_M)

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
SHARED_OPTIMIZER := $(SEQ1024_html) $(SEQ1024_bbox) ++trainer.train.optimizer.lr=1e-4 \
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


# 임시로 사용
ONLY_STRUCTURE = ++trainer.trainer.only_structure=True





############## AB : HTML VS OTSL ##############
# 	$(PTN_40000_M) $(FIN_20000_M) $(SYN_PUB_10000_M) $(SYN_FIN_10000_M) $(SYN_MKT_10000_M) $(SYN_SPS_10000_M) 

DATA = $(DATA_MULTI) $(DATA_AB) # MINIPUBTABNET_M   DATA_AB

# make experiments/ab_base_html/.done_finetune
EXP_ab_base_html := $(COMMONS_BASE) $(HTML) $(BBOX) $(DATA) $(ONLY_STRUCTURE)\

# make experiments/ab_base_otsl/.done_finetune
EXP_ab_base_otsl := $(COMMONS_BASE) $(HTML) $(BBOX) $(DATA) $(ONLY_STRUCTURE)\
					$(OTSL_MODE) 





############## AB : 데이터셋 ##############

# make experiments/ab_base/.done_finetune
EXP_ab_base := $(COMMONS_BASE) $(HTML) $(BBOX) $(DATA)
				

# make experiments/ab_base_case1/.done_finetune
EXP_ab_base_case1 := $(COMMONS_BASE) $(HTML) $(BBOX) $(DATA)\
						$(IX_mvp3_case1_250411)

# make experiments/ab_base_case3/.done_finetune
EXP_ab_base_case3 := $(COMMONS_BASE) $(HTML) $(BBOX) $(DATA)\
						$(IX_mvp3_case3_250411)

# make experiments/ab_base_gen_image/.done_finetune
EXP_ab_base_gen_image := $(COMMONS_BASE) $(HTML) $(BBOX) $(DATA)\
						$(GEN_IMAGE_1000)

# make experiments/ab_base_gen_formula/.done_finetune
EXP_ab_base_gen_formula := $(COMMONS_BASE) $(HTML) $(BBOX) $(DATA)\
						$(GEN_FORMULA_1000)




############## AB : MIX LOSS ##############

# make experiments/ab_base_mix_loss/.done_finetune
# DATA = $(DATA_MULTI) $(SYN_MKT_10000_M) # MINIPUBTABNET_M   DATA_AB
EXP_ab_base_mix_loss := $(COMMONS_BASE) $(HTML) $(BBOX) $(DATA)\
						$(USE_MIX_LOSS)

						


############## MVP2.5 : COMPARE ##############
EPOCH = $(EPOCH30)
BATCH = $(BATCH32) # 92 76 64 48 32
LOCK_MTIM_0 = ++trainer.trainer.freeze_beit_epoch=0
DATA = $(DATA_MULTI) $(DATA_AB)\
		$(GEN_FORMULA_1000) $(GEN_IMAGE_1000)\
		

# make experiments/ab_large_case1_ft/.done_finetune
EXP_ab_large_case1_ft := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA) $(LOCK_MTIM_0) $(BATCH) $(EPOCH)\
							$(IX_mvp3_case1_250411)\
							$(FT_MODE)\
							$(USE_MIX_LOSS)

# make experiments/ab_large_case3_ft/.done_finetune
EXP_ab_large_case3_ft := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA) $(LOCK_MTIM_0) $(BATCH) $(EPOCH)\
							$(IX_mvp3_case3_250411)\
							$(FT_MODE)\
							$(USE_MIX_LOSS)





# DATA_TEST = $(DATA_MULTI) $(SYN_PUB_10000_M)

############## AB : ERROR VS CLEAN ##############
# Clean Dataset이 성능 더 좋음 👍

# make experiments/ab_base_error/.done_finetune
EXP_ab_base_error := $(COMMONS_BASE) $(HTML) $(BBOX) $(DATA_AB_ERROR)

# make experiments/ab_base_clean/.done_finetune
EXP_ab_base_clean := $(COMMONS_BASE) $(HTML) $(BBOX) $(DATA_AB_CLEAN)


############## AB : MIX LOSS ##############
# Mix Loss 성능 더 좋음 👍

# make experiments/ab_base_mix_loss_clean/.done_finetune
EXP_ab_base_mix_loss_clean := $(COMMONS_BASE) $(HTML) $(BBOX) $(DATA_AB_CLEAN)\
						$(USE_MIX_LOSS)



############## AB : MIX LOSS / OTSL ##############

# make experiments/ab_base_otsl_mix_loss/.done_finetune
EXP_ab_base_otsl_mix_loss := $(COMMONS_BASE) $(HTML) $(BBOX) $(DATA_AB_CLEAN)\
						$(USE_MIX_LOSS) $(OTSL_MODE)




############## AB : 추가 데이터셋 ##############


# make experiments/ab_base_case1_clean/.done_finetune
EXP_ab_base_case1_clean := $(COMMONS_BASE) $(HTML) $(BBOX) $(DATA_AB_CLEAN)\
						$(IX_mvp3_case1_250411)

# make experiments/ab_base_case3_clean/.done_finetune
EXP_ab_base_case3_clean := $(COMMONS_BASE) $(HTML) $(BBOX) $(DATA_AB_CLEAN)\
						$(IX_mvp3_case3_250411)


############## AB : HTML VS OTSL | (100K) ###############

# make experiments/ab_base_html_clean_only_structure/.done_finetune
EXP_ab_base_html_clean_only_structure := $(COMMONS_BASE) $(HTML) $(BBOX) $(DATA_AB_CLEAN) $(ONLY_STRUCTURE)\

# make experiments/ab_base_otsl_clean_only_structure/.done_finetune
EXP_ab_base_otsl_clean_only_structure := $(COMMONS_BASE) $(HTML) $(BBOX) $(DATA_AB_CLEAN) $(ONLY_STRUCTURE)\


############## AB : HTML VS OTSL | (500K) ###############
# make experiments/ab_large_html_500K/.done_finetune
EXP_ab_large_html_500K := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA_AB_CLEAN_500K)\
					$(BATCH32) $(EPOCH30)

# make experiments/ab_large_otsl_500K/.done_finetune
EXP_ab_large_otsl_500K := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA_AB_CLEAN_500K)\
					$(BATCH32) $(EPOCH30) $(OTSL_MODE) 

# make experiments/ab_large_mix_loss_500K/.done_finetune
EXP_ab_large_mix_loss_500K := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA_AB_CLEAN_500K)\
					$(BATCH32) $(EPOCH30) $(USE_MIX_LOSS)

# make experiments/ab_large_otsl_mix_loss_500K/.done_finetune
EXP_ab_large_otsl_mix_loss_500K := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA_AB_CLEAN_500K)\
					$(BATCH32) $(EPOCH30) $(USE_MIX_LOSS) $(OTSL_MODE) 

DATA_TEST = $(DATA_MULTI) $(PTN_40000_M)




EXP_ab_large_otsl_1M_no_fin := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA_1M_NO_FIN)\
					$(BATCH24) $(EPOCH25) $(OTSL_MODE)


# make experiments/ab_large_otsl_mix_loss_1M/.done_finetune
EXP_ab_large_otsl_mix_loss_1M := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA_1M_NO_FIN)\
					$(BATCH24) $(EPOCH30) $(OTSL_MODE) $(USE_MIX_LOSS)


# 🔥🔥🔥🔥 MVP 2.7 🔥🔥🔥🔥
EXP_large_otsl_mix_AUG_1M_add_ix := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA_1M_IX)\
					$(BATCH24) $(EPOCH30) $(OTSL_MODE) $(USE_MIX_LOSS) $(USE_AUG)





########################### MVP3.0 #######################


# MVP2.7 모델에 Gradient Accumulation만 적용함.
# 현재 30epoch 까지만 학습 진행함.
# make experiments/large_otsl_mix_AUG_1M_add_ix_grad_accum/.done_finetune

# 0~29 epoch : mvp2.7 데이터
# 30~47 epoch : mvp3.0 추가 데이터 학습

snapshot_html = ++trainer.trainer.snapshot_html="epoch29_snapshot_html.pt" 
snapshot_bbox = ++trainer.trainer.snapshot_bbox="epoch29_snapshot_bbox.pt"
snapshot_beit = ++trainer.trainer.beit_pretrained_weights=null
snapshot = $(snapshot_html) $(snapshot_bbox) $(snapshot_beit)

# MVP 3.0 🔥
EXP_large_otsl_mix_AUG_1M_add_ix_grad_accum := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA_1M_IX_MVP30)\
					$(BATCH32) $(EPOCH48) $(OTSL_MODE) $(USE_MIX_LOSS) $(USE_AUG) $(snapshot)




# MVP 3.0 recovery
# make experiments/recover_mvp3/.done_finetune

EXP_recover_mvp3 := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA_1M_IX_MVP30)\
					$(BATCH32) $(EPOCH48) $(OTSL_MODE) $(USE_MIX_LOSS) $(USE_AUG) $(snapshot)




########################### MVP4.0 #######################

# make experiments/test_break_model/.done_finetune
EXP_test_break_model := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA_1M_IX)\
					$(BATCH32) $(EPOCH48) $(OTSL_MODE) $(USE_MIX_LOSS) $(USE_AUG) $(snapshot)


# make experiments/mvp4_add_case1/.done_finetune
EXP_mvp4_add_case1 := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA_1M_IX_MVP30)\
					$(BATCH32) $(EPOCH31) $(OTSL_MODE) $(USE_MIX_LOSS) $(USE_AUG) $(snapshot) \
					$(IX_mvp4_case1_200)

# make experiments/mvp4_add_case2/.done_finetune
EXP_mvp4_add_case2 := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA_1M_IX_MVP30)\
					$(BATCH32) $(EPOCH31) $(OTSL_MODE) $(USE_MIX_LOSS) $(USE_AUG) $(snapshot) \
					$(IX_mvp4_case2_200)

# make experiments/mvp4_add_case3_inj/.done_finetune
EXP_mvp4_add_case3_inj := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA_1M_IX_MVP30)\
					$(BATCH32) $(EPOCH31) $(OTSL_MODE) $(USE_MIX_LOSS) $(USE_AUG) $(snapshot) \
					$(IX_mvp4_case3_injection_molding)



# make experiments/mvp4_break_test_mvp3/.done_finetune
EXP_mvp4_break_test_mvp3 := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA_1M_IX_MVP30)\
					$(BATCH32) $(EPOCH48) $(OTSL_MODE) $(USE_MIX_LOSS) $(USE_AUG) $(snapshot)


# make experiments/mvp4_break_test_case3/.done_finetune
EXP_mvp4_break_test_case3 := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA_1M_IX_MVP30)\
					$(BATCH32) $(EPOCH48) $(OTSL_MODE) $(USE_MIX_LOSS) $(USE_AUG)\
					$(IX_mvp4_case3_injection_molding)



# make experiments/mvp4_break_test_case3_snap29/.done_finetune
snapshot_html_29 = ++trainer.trainer.snapshot_html="epoch29_snapshot_html.pt" 
snapshot_bbox_29 = ++trainer.trainer.snapshot_bbox="epoch29_snapshot_bbox.pt"
snapshot_29 = $(snapshot_html_29) $(snapshot_bbox_29) $(snapshot_beit)

EXP_mvp4_break_test_case3_snap29 := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA_1M_IX_MVP30)\
					$(BATCH32) $(EPOCH48) $(OTSL_MODE) $(USE_MIX_LOSS) $(USE_AUG)\
					$(IX_mvp4_case3_injection_molding) $(snapshot_29)






########################### MVP4.0 : Max Token 증가 테스트 #######################
# HTML : 512 > 1024
# BBOX : 1024 > 2048
# Batch : 16
# Batch Accumulation : HTML 6 / BBOX 4

# make experiments/mvp4_max_token/.done_finetune
EXP_mvp4_max_token := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA_1M_IX_MVP30)\
					$(BATCH16) $(EPOCH48) $(OTSL_MODE) $(USE_MIX_LOSS) $(USE_AUG)\
					$(IX_mvp4_case3_injection_molding)\
					$(SEQ1024_html) $(SEQ2048_bbox)





########################### SharedEncoder_DualDecoder Experiments #######################

# Use SharedEncoder_DualDecoder architecture
MODEL_SHARED_ENCODER_DUAL_DECODER = model=sharedencoder_dualdecoder
TRAINER_SHARED = trainer=table_mix_dualdecoder

# SharedEncoder_DualDecoder specific settings
SHARED_ARCH_LARGE = $(MTIM_LARGE) $(MODEL_SHARED_ENCODER_DUAL_DECODER) $(D4)

COMMONS_SHARED_LARGE := $(VOCAB_MIX) $(USE_HTML)\
			$(WEIGHTS_mtim_2m_large) $(SHARED_ARCH_LARGE)\
			$(EPOCH)\
			$(BATCH)\
			$(LABEL_MIX) $(AUG_RESIZE_NORM)\
			$(TRAINER_SHARED) $(I448)\
			$(LOCK_MTIM_4)

# SharedEncoder_DualDecoder experiment
EXP_shared_encoder_dual_decoder := $(COMMONS_SHARED_LARGE) $(SHARED_OPTIMIZER) $(DATA_AB_CLEAN)\
						$(USE_MIX_LOSS) $(OTSL_MODE)

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




LR_seq_html = trainer/train/lr_scheduler_html=sequential
LR_seq_bbox = trainer/train/lr_scheduler_bbox=sequential

HTML := $(SEQ512_html)  $(OPT_ADAMW_HTML) $(OPT_WD5e2_HTML) $(LR_2e4_html) $(LR_seq_html)
BBOX := $(SEQ1024_bbox) $(OPT_ADAMW_BBOX) $(OPT_WD5e2_BBOX) $(LR_2e4_bbox) $(LR_seq_bbox)\
		$(GRAD_CLIP12)

# make experiments/mvp4_lr_sequential_try2/.done_finetune
EXP_mvp4_lr_sequential_try2 := $(COMMONS_LARGE) $(HTML) $(BBOX) $(DATA_1M_IX_MVP40)\
					$(BATCH16) $(EPOCH48) $(OTSL_MODE) $(USE_MIX_LOSS) $(USE_AUG)\
					$(SEQ1024_html) $(SEQ2048_bbox)


# DATA_1M_IX_MVP30
# DATA_MINI






.PHONY: finetune_all
finetune_all:\
experiments/mvp30_adding_korean_table_in_ssp_E24/.done_finetune\



# make finetune_all