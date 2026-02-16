PYTHON ?= python
MAVEN_SCRIPT := /home/bizon/research/terraforming/scripts/download_maven_kp_api.py
MAVEN_OUT ?= /home/bizon/research/terraforming/data/maven/sdc_api_full

# Baseline overlap window from docs/implementation/01_baseline_mars_system.md
BASELINE_START ?= 2014-10-19
BASELINE_END ?= 2016-08-17
TIMEOUT ?= 180

.PHONY: help \
	maven-kp-insitu maven-kp-insitu-dry \
	maven-kp-iuvs maven-kp-iuvs-dry \
	maven-ngi-l2l3 maven-ngi-l2l3-dry \
	maven-euv-minute maven-euv-minute-dry \
	maven-swi-core maven-swi-core-dry \
	maven-sta-core maven-sta-core-dry \
	maven-sep-cal maven-sep-cal-dry \
	maven-sep-full maven-sep-full-dry \
	maven-lpw-l2-core maven-lpw-l2-core-dry \
	maven-lpw-full maven-lpw-full-dry \
	maven-mag-full maven-mag-full-dry \
	maven-mag1s maven-mag1s-dry \
	maven-swea-min maven-swea-min-dry \
	maven-swea-recommended maven-swea-recommended-dry \
	maven-swea-full maven-swea-full-dry

help:
	@echo "Available targets:"
	@echo "  maven-kp-insitu             - KP insitu baseline pull"
	@echo "  maven-kp-iuvs               - KP IUVS baseline pull"
	@echo "  maven-ngi-l2l3              - NGIMS L2+L3 pull"
	@echo "  maven-euv-minute            - EUV L3 minute pull"
	@echo "  maven-swi-core              - SWIA core plans (finearc3d,finesvy3d,onboardsvymom,onboardsvyspec)"
	@echo "  maven-sta-core              - STATIC core plans (c6,c8,ca,d0,d1,ce,cf)"
	@echo "  maven-sep-cal               - SEP calibrated survey pull (s1/s2)"
	@echo "  maven-sep-full              - SEP raw+calibrated+ancillary pull"
	@echo "  maven-lpw-l2-core           - LPW L2 core pull (lpiv,lpnt,mrgscpot,we12,wn,wspec*)"
	@echo "  maven-lpw-full              - LPW full pull (L0b + L2 plans listed by MAVEN SDC)"
	@echo "  maven-mag-full              - MAG unfiltered L2 pull (very large)"
	@echo "  maven-mag1s                 - Reduced MAG pull (pc1s,ss1s)"
	@echo "  maven-swea-min              - SWEA minimal pull (svyspec)"
	@echo "  maven-swea-recommended      - SWEA recommended pull (svyspec,svy3d)"
	@echo "  maven-swea-full             - SWEA full pull (all major plans)"
	@echo ""
	@echo "Dry-run variants:"
	@echo "  maven-kp-insitu-dry, maven-kp-iuvs-dry, maven-ngi-l2l3-dry,"
	@echo "  maven-euv-minute-dry, maven-swi-core-dry, maven-sta-core-dry,"
	@echo "  maven-sep-cal-dry, maven-sep-full-dry, maven-lpw-l2-core-dry,"
	@echo "  maven-lpw-full-dry,"
	@echo "  maven-mag-full-dry, maven-mag1s-dry, maven-swea-min-dry,"
	@echo "  maven-swea-recommended-dry, maven-swea-full-dry"
	@echo ""
	@echo "Overridable vars:"
	@echo "  BASELINE_START=$(BASELINE_START)"
	@echo "  BASELINE_END=$(BASELINE_END)"
	@echo "  MAVEN_OUT=$(MAVEN_OUT)"
	@echo "  TIMEOUT=$(TIMEOUT)"

maven-kp-insitu:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument kp \
		--levels insitu \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT)

maven-kp-insitu-dry:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument kp \
		--levels insitu \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT) \
		--dry-run

maven-kp-iuvs:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument kp \
		--levels iuvs \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT)

maven-kp-iuvs-dry:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument kp \
		--levels iuvs \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT) \
		--dry-run

maven-ngi-l2l3:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument ngi \
		--levels l2,l3 \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT)

maven-ngi-l2l3-dry:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument ngi \
		--levels l2,l3 \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT) \
		--dry-run

maven-euv-minute:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument euv \
		--plans minute \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT)

maven-euv-minute-dry:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument euv \
		--plans minute \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT) \
		--dry-run

maven-swi-core:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument swi \
		--plans finearc3d,finesvy3d,onboardsvymom,onboardsvyspec \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT)

maven-swi-core-dry:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument swi \
		--plans finearc3d,finesvy3d,onboardsvymom,onboardsvyspec \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT) \
		--dry-run

maven-sta-core:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument sta \
		--plans c6,c8,ca,d0,d1,ce,cf \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT)

maven-sta-core-dry:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument sta \
		--plans c6,c8,ca,d0,d1,ce,cf \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT) \
		--dry-run

maven-sep-cal:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument sep \
		--plans s1-cal-svy-full,s2-cal-svy-full \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--sep-chunk-months 1 \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT)

maven-sep-cal-dry:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument sep \
		--plans s1-cal-svy-full,s2-cal-svy-full \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--sep-chunk-months 1 \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT) \
		--dry-run

maven-sep-full:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument sep \
		--plans s1-raw-svy-full,s2-raw-svy-full,s1-cal-svy-full,s2-cal-svy-full,anc \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--sep-chunk-months 1 \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT)

maven-sep-full-dry:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument sep \
		--plans s1-raw-svy-full,s2-raw-svy-full,s1-cal-svy-full,s2-cal-svy-full,anc \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--sep-chunk-months 1 \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT) \
		--dry-run

maven-lpw-l2-core:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument lpw \
		--plans lpiv,lpnt,mrgscpot,we12,wn,wspecact,wspecpas \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--lpw-chunk-months 1 \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT)

maven-lpw-l2-core-dry:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument lpw \
		--plans lpiv,lpnt,mrgscpot,we12,wn,wspecact,wspecpas \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--lpw-chunk-months 1 \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT) \
		--dry-run

maven-lpw-full:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument lpw \
		--plans act,adr,atr,euv,hsbmhf,hsbmlf,hsbmmf,hsk,pas,spechfact,spechfpas,speclfact,speclfpas,specmfact,specmfpas,swp1,sw2,lpiv,lpnt,mrgscpot,we12,we12bursthf,we12burstlf,we12burstmf,wn,wspecact,wspecpas \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--lpw-chunk-months 1 \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT)

maven-lpw-full-dry:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument lpw \
		--plans act,adr,atr,euv,hsbmhf,hsbmlf,hsbmmf,hsk,pas,spechfact,spechfpas,speclfact,speclfpas,specmfact,specmfpas,swp1,sw2,lpiv,lpnt,mrgscpot,we12,we12bursthf,we12burstlf,we12burstmf,wn,wspecact,wspecpas \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--lpw-chunk-months 1 \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT) \
		--dry-run

maven-mag-full:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument mag \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--plans all \
		--mag-chunk-days 7 \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT)

maven-mag-full-dry:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument mag \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--plans all \
		--mag-chunk-days 7 \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT) \
		--dry-run

maven-mag1s:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument mag1s \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--plans pc1s,ss1s \
		--mag1s-chunk-days 7 \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT)

maven-mag1s-dry:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument mag1s \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--plans pc1s,ss1s \
		--mag1s-chunk-days 7 \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT) \
		--dry-run

maven-swea-min:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument swe \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--plans svyspec \
		--swe-chunk-days 7 \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT)

maven-swea-min-dry:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument swe \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--plans svyspec \
		--swe-chunk-days 7 \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT) \
		--dry-run

maven-swea-recommended:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument swe \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--plans svyspec,svy3d \
		--swe-chunk-days 7 \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT)

maven-swea-recommended-dry:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument swe \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--plans svyspec,svy3d \
		--swe-chunk-days 7 \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT) \
		--dry-run

maven-swea-full:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument swe \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--plans arcpad,svypad,arc3d,svy3d,svyspec \
		--swe-chunk-days 7 \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT)

maven-swea-full-dry:
	$(PYTHON) $(MAVEN_SCRIPT) \
		--instrument swe \
		--start-date $(BASELINE_START) \
		--end-date $(BASELINE_END) \
		--plans arcpad,svypad,arc3d,svy3d,svyspec \
		--swe-chunk-days 7 \
		--output-dir $(MAVEN_OUT) \
		--timeout $(TIMEOUT) \
		--dry-run

