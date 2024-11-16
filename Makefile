ARCHIVE_NAME=xvojta09.zip
TARGETS=docs src results data main.py build.sh README.md config.yaml requirements.txt

zip: docs/documentation.pdf $(TARGETS) 
	zip -r $(ARCHIVE_NAME) $^

clean: $(ARCHIVE_NAME)
	rm $(ARCHIVE_NAME)
