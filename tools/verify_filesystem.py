"""Run the same filesystem acceptance selection on every native CI environment."""

from __future__ import annotations

import unittest

SUITES = (
    "tests.test_checker_process",
    "tests.test_filesystem",
    "tests.test_filesystem_process",
    "tests.test_app_config",
    "tests.test_bootstrap_filesystem.BootstrapReadTests",
    "tests.test_bootstrap_filesystem.BootstrapFilesystemTests",
    "tests.test_core_tools",
    "tests.test_entrypoint.ResourceCleanupTests",
    "tests.test_presentation",
    "tests.test_package_system.PackageTransactionTests",
    "tests.test_package_sources",
    "tests.test_catalog_publication",
    "tests.test_evidence_log",
    "tests.test_session_journal",
    "tests.test_workspace_trust",
    "tests.test_workspace_transactions",
    "tests.test_plugins.PluginRetirementTests",
    "tests.test_hot_plugins.SourceCaptureTests",
    "tests.test_release",
    "tests.test_build_portable.PortableBuildTests",
    "tests.test_build_portable.PortableSourceTests",
    "tests.test_release_folder",
    "tests.test_gepa_engine",
    "tests.test_optimize_chat_prompt",
    "tests.test_self_harness.ExperimentPublicationTests",
    "tests.test_self_harness.CandidatePromotionTests",
    "tests.test_self_harness.SelfHarnessTests",
    "tests.test_workflow_stress.WorkflowCleanupTests",
    "tests.test_package_recovery",
    "tests.test_plugin_filesystem_process",
    "tests.test_plugin_memory",
    "tests.test_plugin_skills.SkillStoreTests",
)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromNames(SUITES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
