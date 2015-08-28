
.NOTPARALLEL: %.sift


all: train-done
	

IMAGES := $(sort $(wildcard database/train/*/*.jpg))
IMAGES_TEST := $(sort $(wildcard database/test/*/*.jpg))
FEATS_SIFT := $(patsubst %.jpg,%.sift,$(IMAGES))
FEATS_SIFT_TEST := $(patsubst %.jpg,%.sift,$(IMAGES_TEST))

train-done: local/all.sift  local/all.labels local/test/all.sift local/test/all.labels
	./main.py $(word 1, $^) $(word 2, $^) $(word 3, $^) $(word 4, $^)

%.sift : %.jpg
	./feature_extractor.py $< $@

feats/% : %
	echo $(IMAGES)
	echo $(FEATS_SIFT)

feats: local/all.sift
feats-test: local/test/all.sift

local/all.sift: $(IMAGES) $(FEATS_SIFT) 
	echo "$(FEATS_SIFT)" > $@

local/test/all.sift: $(IMAGES_TEST) $(FEATS_SIFT_TEST) 
	mkdir -p local/test
	echo "$(FEATS_SIFT_TEST)" > $@


labels-train: local/all.labels
labels-test: local/test/all.labels

local/all.labels:
	echo "$(notdir $(shell dirname $(IMAGES)))" > $@

local/test/all.labels:
	mkdir -p local/test
	echo "$(notdir $(shell dirname $(IMAGES_TEST)))" > $@


clean-local:
	rm -rf local/*	

clean-feats:
	rm -f $(FEATS_SIFT) $(FEATS_SIFT_TEST) 
