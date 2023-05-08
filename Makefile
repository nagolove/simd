# GNU Make workspace makefile autogenerated by Premake

ifndef config
  config=default
endif

ifndef verbose
  SILENT = @
endif

ifeq ($(config),default)
  simd_1_config = default
endif

PROJECTS := simd_1

.PHONY: all clean help $(PROJECTS) 

all: $(PROJECTS)

simd_1:
ifneq (,$(simd_1_config))
	@echo "==== Building simd_1 ($(simd_1_config)) ===="
	@${MAKE} --no-print-directory -C . -f simd_1.make config=$(simd_1_config)
endif

clean:
	@${MAKE} --no-print-directory -C . -f simd_1.make clean

help:
	@echo "Usage: make [config=name] [target]"
	@echo ""
	@echo "CONFIGURATIONS:"
	@echo "  default"
	@echo ""
	@echo "TARGETS:"
	@echo "   all (default)"
	@echo "   clean"
	@echo "   simd_1"
	@echo ""
	@echo "For more information, see https://github.com/premake/premake-core/wiki"