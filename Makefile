all:
	cargo build --release
	ln -s target/release/FeatureReportGenerator ./feat_report_gen.binary

clean:
	cargo clean
	rm ./feat_report_gen.binary