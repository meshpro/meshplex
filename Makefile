VERSION=$(shell python3 -c "import voropy; print(voropy.__version__)")

default:
	@echo "\"make publish\"?"

tag:
	# Make sure we're on the master branch
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	@echo "Tagging v$(VERSION)..."
	git tag v$(VERSION)
	git push --tags

README.rst: README.md
	pandoc README.md -o README.rst
	python3 setup.py check -r -s || exit 1

upload: setup.py README.rst
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	rm -f dist/*
	python3 setup.py bdist_wheel --universal
	gpg --detach-sign -a dist/*
	twine upload dist/*

publish: tag upload

clean:
	rm -f README.rst
