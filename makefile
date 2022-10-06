SHELL = /bin/bash

project_dependencies ?= $(addprefix $(project_root)/, emissor \
    cltl-combot \
    cltl-requirements )

git_remote ?= https://github.com/leolani

include util/make/makefile.base.mk
include util/make/makefile.component.mk
include util/make/makefile.py.base.mk
include util/make/makefile.git.mk


build: resources/face_models/models.lock


resources/face_models/models.lock:
	mkdir -p resources/face_models
	wget -qO- https://surfdrive.surf.nl/files/index.php/s/Qx80CsSNJUgebUg/download | tar xvz -C resources/face_models
	touch resources/face_models/models.lock


clean:
	rm -rf resources/face_models
