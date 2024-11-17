ARCHIVE_NAME=xvojta09.zip
TARGETS=docs src results data main.py build.sh README.md config.yaml requirements.txt

zip: $(TARGETS) 
	zip -r $(ARCHIVE_NAME) $^ --exclude "**/__pycache__/*"

clean: $(ARCHIVE_NAME)
	rm $(ARCHIVE_NAME)
