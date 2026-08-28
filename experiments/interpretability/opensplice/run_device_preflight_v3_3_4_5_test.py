#!/usr/bin/env python3
"""CPU-only contract tests for the v3.3.4.5 external preflight."""

from __future__ import annotations

from contextlib import contextmanager
import ast
import copy
import errno
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import socket
import stat
import tempfile
import time
import unittest
from unittest import mock


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))
import launch_encoder_skip_ood_sidecar_v3_3_4_5 as launcher  # pylint: disable=g-import-not-at-top
import run_device_preflight_v3_3_4_5 as preflight  # pylint: disable=g-import-not-at-top


def _environment(root: Path, role: str = 'external_preflight') -> dict[str, str]:
  return {
      'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
      'JAX_ENABLE_COMPILATION_CACHE': 'false',
      'CUDA_CACHE_DISABLE': '1',
      'ALPHAGENOME_V3_3_4_5_CACHE_ROLE': role,
      'ALPHAGENOME_V3_3_4_5_CACHE_ROOT': str(root.resolve()),
      'TRITON_CACHE_DIR': str((root / 'triton').resolve()),
      'XDG_CACHE_HOME': str((root / 'xdg').resolve()),
  }


class DevicePreflightV3345Test(unittest.TestCase):

  @contextmanager
  def _roots(self, base: Path):
    external = base / 'external-cache'
    preflight_root = base / 'preflight'
    external.mkdir(mode=0o700)
    (external / 'triton').mkdir(mode=0o700)
    (external / 'xdg').mkdir(mode=0o700)
    old_roots = dict(preflight.bootstrap.PUBLICATION_ROOTS)
    old_cache = preflight.bootstrap.PREFLIGHT_KERNEL_CACHE_DIR
    old_preflight = preflight.bootstrap.PREFLIGHT_DIR
    old_local_preflight = preflight.PREFLIGHT_DIR
    preflight.bootstrap.PUBLICATION_ROOTS['external_cache'] = external
    preflight.bootstrap.PUBLICATION_ROOTS['external_preflight'] = preflight_root
    preflight.bootstrap.PREFLIGHT_KERNEL_CACHE_DIR = external
    preflight.bootstrap.PREFLIGHT_DIR = preflight_root
    preflight.PREFLIGHT_DIR = preflight_root
    for value in (
        preflight.bootstrap._PUBLICATION_SUCCESS,
        preflight.bootstrap._PUBLICATION_TEMP_ORPHANS,
        preflight.bootstrap._PUBLICATION_UNCERTAIN_FINALS,
        preflight.bootstrap._PUBLICATION_PREEXISTING,
        preflight.bootstrap._PUBLICATION_UNBINDABLE_FAILURES,
    ):
      value.clear()
    preflight.bootstrap._PUBLICATION_DIRECTORIES.clear()
    preflight.bootstrap._PUBLICATION_ORDINAL = 0
    try:
      with mock.patch.dict(os.environ, _environment(external), clear=True):
        yield external, preflight_root
    finally:
      preflight.bootstrap.PUBLICATION_ROOTS.clear()
      preflight.bootstrap.PUBLICATION_ROOTS.update(old_roots)
      preflight.bootstrap.PREFLIGHT_KERNEL_CACHE_DIR = old_cache
      preflight.bootstrap.PREFLIGHT_DIR = old_preflight
      preflight.PREFLIGHT_DIR = old_local_preflight
      preflight.bootstrap._PUBLICATION_DIRECTORIES.clear()

  def test_dry_plan_is_jax_only_and_creates_nothing(self):
    plan = preflight.build_dry_run_plan()
    self.assertEqual((plan['model_calls'], plan['jit_calls']), (0, 0))
    self.assertFalse(plan['scientific_output_created'])
    self.assertEqual(plan['sole_attempt_number'], 0)
    self.assertTrue(plan['atomic_publication_probe_required'])

  def test_three_way_preflight_version_proof_rejects_each_drift(self):
    module = preflight.bootstrap
    source = Path(preflight.__file__)
    relative = source.relative_to(module._REPO).as_posix()  # pylint: disable=protected-access
    frozen = {
        'preflight_script_version': module.PREFLIGHT_SCRIPT_VERSION,
        'file_sha256': {relative: hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    proof = module.validate_preflight_version_contract(frozen)
    self.assertTrue(proof['freeze_equals_bootstrap'])
    self.assertTrue(proof['bootstrap_equals_producer_literal'])
    self.assertTrue(proof['validated_before_allocation_or_registration'])

    wrong_freeze = copy.deepcopy(frozen)
    wrong_freeze['preflight_script_version'] = 'v3.3.4.5'
    with self.assertRaisesRegex(ValueError, 'three-way proof'):
      module.validate_preflight_version_contract(wrong_freeze)
    with mock.patch.object(module, 'PREFLIGHT_SCRIPT_VERSION', 'v3.3.4.5'):
      with self.assertRaisesRegex(ValueError, 'three-way proof'):
        module.validate_preflight_version_contract(frozen)

    with tempfile.TemporaryDirectory() as directory:
      temporary = Path(directory)
      producer = temporary / 'run_device_preflight_v3_3_4_5.py'
      producer.write_text(
          "PREFLIGHT_SCRIPT_VERSION = 'v3.3.4.5'\n", encoding='utf-8'
      )
      producer_frozen = {
          'preflight_script_version': module.PREFLIGHT_SCRIPT_VERSION,
          'file_sha256': {
              producer.name: hashlib.sha256(producer.read_bytes()).hexdigest()
          },
      }
      with (
          mock.patch.object(module, '_HERE', temporary),
          mock.patch.object(module, '_REPO', temporary),
          self.assertRaisesRegex(ValueError, 'three-way proof'),
      ):
        module.validate_preflight_version_contract(producer_frozen)

  def test_gate_a_and_child_version_proof_precede_every_production_allocation(self):
    parent = Path(launcher.__file__).read_text(encoding='utf-8')
    parent_positions = [parent.index(token) for token in (
        'path_absence = bootstrap.validate_gate_a_path_absence()',
        'gate_a = bootstrap.validate_freeze()',
        "_allocate_cache(\n      bootstrap.PREFLIGHT_KERNEL_CACHE_DIR",
        'subprocess.run(',
    )]
    self.assertEqual(parent_positions, sorted(parent_positions))

    child = Path(preflight.__file__).read_text(encoding='utf-8')
    child_main = child[child.index('def run_preflight()'):]
    child_positions = [child_main.index(token) for token in (
        'freeze = _validate_authorized_freeze()',
        'pre_import = _register_external_cache()',
        'probe = _formal_publication_probe()',
        '_allocate_preflight_root()',
        'observation = _collect_observation(pre_import)',
    )]
    self.assertEqual(child_positions, sorted(child_positions))
    authorized = child[
        child.index('def _validate_authorized_freeze()'):
        child.index('def _cache_hit_evidence')
    ]
    self.assertLess(
        authorized.index('validate_preflight_version_contract(frozen)'),
        authorized.index(
            'validate_prior_v3_3_4_3_consumed_preflight_prefix()'
        ),
    )

    with (
        mock.patch.object(
            preflight, '_validate_authorized_freeze',
            side_effect=ValueError('three-way version mismatch'),
        ),
        mock.patch.object(preflight, '_register_external_cache') as register,
        mock.patch.object(preflight, '_formal_publication_probe') as probe,
        mock.patch.object(preflight, '_allocate_preflight_root') as allocate,
        self.assertRaisesRegex(ValueError, 'version mismatch'),
    ):
      preflight.run_preflight()
    register.assert_not_called()
    probe.assert_not_called()
    allocate.assert_not_called()

  def test_consumed_prefix_embedded_object_rejects_every_leaf_and_key_drift(self):
    module = preflight.bootstrap
    prefix = module.validate_prior_v3_3_4_3_consumed_preflight_prefix()
    binding = module.canonical_content_binding(prefix)

    def leaves(value, path=()):
      if isinstance(value, dict):
        for key, item in value.items():
          yield from leaves(item, path + (key,))
      elif isinstance(value, list):
        for index, item in enumerate(value):
          yield from leaves(item, path + (index,))
      else:
        yield path, value

    def change(value):
      if value is None:
        return 'drift'
      if isinstance(value, bool):
        return not value
      if isinstance(value, int):
        return value + 1
      return f'{value}-drift'

    def assign(value, path, replacement):
      cursor = value
      for token in path[:-1]:
        cursor = cursor[token]
      cursor[path[-1]] = replacement

    with mock.patch.object(
        module, 'validate_prior_v3_3_4_3_consumed_preflight_prefix',
        return_value=prefix,
    ):
      module.validate_recorded_prior_v3_3_4_3_consumed_preflight_prefix(
          prefix, binding
      )
      for path, value in leaves(prefix):
        with self.subTest(path=path):
          drifted = copy.deepcopy(prefix)
          assign(drifted, path, change(value))
          with self.assertRaisesRegex(ValueError, 'object changed'):
            module.validate_recorded_prior_v3_3_4_3_consumed_preflight_prefix(
                drifted, binding
            )
      for altered in (
          {**prefix, 'unexpected': True},
          {key: value for key, value in prefix.items() if key != 'status'},
      ):
        with self.assertRaisesRegex(ValueError, 'object changed'):
          module.validate_recorded_prior_v3_3_4_3_consumed_preflight_prefix(
              altered, binding
          )
      for altered in (
          {**binding, 'sha256': '0' * 64},
          {**binding, 'size_bytes': binding['size_bytes'] + 1},
          {**binding, 'unexpected': True},
          {'sha256': binding['sha256']},
      ):
        with self.assertRaisesRegex(ValueError, 'binding changed'):
          module.validate_recorded_prior_v3_3_4_3_consumed_preflight_prefix(
              prefix, altered
          )

  def test_consumed_prefix_live_sources_lstat_cache_and_all_absences_fail_closed(self):
    module = preflight.bootstrap
    prefix = module.validate_prior_v3_3_4_3_consumed_preflight_prefix()
    with mock.patch.object(module, 'V3_3_4_3_FREEZE_SHA256', '0' * 64):
      with self.assertRaises(ValueError):
        module.validate_prior_v3_3_4_3_consumed_preflight_prefix()
    for relative in module.V3_3_4_3_SOURCE_BINDINGS:
      with self.subTest(source=relative):
        drifted = dict(module.V3_3_4_3_SOURCE_BINDINGS)
        drifted[relative] = '0' * 64
        with mock.patch.object(module, 'V3_3_4_3_SOURCE_BINDINGS', drifted):
          with self.assertRaises(ValueError):
            module.validate_prior_v3_3_4_3_consumed_preflight_prefix()

    root = module.V3_3_4_3_PREDECESSOR_PATHS['external_cache']
    real_lstat = Path.lstat
    for relative in ('.', 'triton', 'xdg'):
      target = root if relative == '.' else root / relative
      for index, label in ((0, 'mode'), (1, 'ino'), (2, 'dev'),
                           (3, 'nlink'), (6, 'size')):
        with self.subTest(path=relative, field=label):
          def altered_lstat(path, *, _target=target, _index=index):
            observed = real_lstat(path)
            if path.resolve() != _target.resolve():
              return observed
            values = list(observed)
            values[_index] = values[_index] + 1
            return os.stat_result(values)
          with mock.patch.object(Path, 'lstat', new=altered_lstat):
            with self.assertRaises(ValueError):
              module.validate_prior_v3_3_4_3_consumed_preflight_prefix()

    cache = prefix['cache_tree_binding']
    for altered in (
        {**cache, 'file_count': 1, 'files': {'extra': {'sha256': '0' * 64}}},
        {**cache, 'directory_count': 2, 'directory_paths': ['.', 'triton']},
        {**cache, 'tree_sha256': '0' * 64},
    ):
      with mock.patch.object(module, 'cache_output_tree_binding', return_value=altered):
        with self.assertRaisesRegex(ValueError, 'cache binding'):
          module.validate_prior_v3_3_4_3_consumed_preflight_prefix()

  def test_v3344_consumed_prefix_exact_binding_and_every_recorded_leaf(self):
    module = preflight.bootstrap
    prefix = module.validate_prior_v3_3_4_4_consumed_preflight_prefix()
    binding = dict(module.V3_3_4_4_CONSUMED_PREFIX_BINDING)
    self.assertEqual(len(prefix), 18)
    self.assertEqual(module.canonical_content_binding(prefix), binding)

    def leaves(value, path=()):
      if isinstance(value, dict):
        for key, item in value.items():
          yield from leaves(item, path + (key,))
      elif isinstance(value, list):
        for index, item in enumerate(value):
          yield from leaves(item, path + (index,))
      else:
        yield path, value

    def mutate(value):
      if value is None:
        return 'drift'
      if isinstance(value, bool):
        return not value
      if isinstance(value, int):
        return value + 1
      return f'{value}-drift'

    with mock.patch.object(
        module, 'validate_prior_v3_3_4_4_consumed_preflight_prefix',
        return_value=prefix,
    ):
      module.validate_recorded_prior_v3_3_4_4_consumed_preflight_prefix(
          prefix, binding
      )
      for path, value in leaves(prefix):
        with self.subTest(path=path):
          changed = copy.deepcopy(prefix)
          cursor = changed
          for token in path[:-1]:
            cursor = cursor[token]
          cursor[path[-1]] = mutate(value)
          with self.assertRaisesRegex(ValueError, 'object changed'):
            module.validate_recorded_prior_v3_3_4_4_consumed_preflight_prefix(
                changed, binding
            )
      for changed_binding in (
          {**binding, 'sha256': '0' * 64},
          {**binding, 'size_bytes': 8654},
          {**binding, 'extra': True},
      ):
        with self.assertRaisesRegex(ValueError, 'binding changed'):
          module.validate_recorded_prior_v3_3_4_4_consumed_preflight_prefix(
              prefix, changed_binding
          )

    with tempfile.TemporaryDirectory() as directory:
      temporary = Path(directory)
      (temporary / 'triton').mkdir()
      (temporary / 'xdg').mkdir()
      (temporary / 'unsafe').symlink_to('triton')
      with self.assertRaisesRegex(ValueError, 'symlink'):
        module.cache_output_tree_binding(temporary)
      (temporary / 'unsafe').unlink()
      os.mkfifo(temporary / 'unsafe')
      with self.assertRaisesRegex(ValueError, 'special'):
        module.cache_output_tree_binding(temporary)

    absent_paths = [
        path for paths in (
            module.V3_3_4_PREDECESSOR_PATHS,
            module.V3_3_4_1_PREDECESSOR_PATHS,
            module.V3_3_4_2_PREDECESSOR_PATHS,
        ) for path in paths.values()
    ] + [
        path for role, path in module.V3_3_4_3_PREDECESSOR_PATHS.items()
        if role != 'external_cache'
    ]
    self.assertEqual(len(absent_paths), 23)
    real_exists = Path.exists
    for target in absent_paths:
      with self.subTest(absent=target.name):
        def exists(path, *, _target=target):
          return path.resolve() == _target.resolve() or real_exists(path)
        with mock.patch.object(Path, 'exists', new=exists):
          with self.assertRaises(FileExistsError):
            module._predecessor_absence_map()  # pylint: disable=protected-access

  def test_v3344_exact_lstat_helpers_reject_every_field_and_bytes(self):
    module = preflight.bootstrap
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / 'root'
      root.mkdir(mode=0o700)
      root_stat = root.lstat()
      expected_directory = (
          root_stat.st_dev, root_stat.st_ino, root_stat.st_nlink,
          root_stat.st_size,
      )
      self.assertEqual(
          module._exact_directory_lstat_row(  # pylint: disable=protected-access
              root, relative='.', expected=expected_directory
          )['path'],
          '.',
      )
      for index, field in enumerate(('st_dev', 'st_ino', 'st_nlink', 'size')):
        with self.subTest(directory_field=field):
          changed = list(expected_directory)
          changed[index] += 1
          with self.assertRaisesRegex(ValueError, 'directory lstat'):
            module._exact_directory_lstat_row(  # pylint: disable=protected-access
                root, relative='.', expected=tuple(changed)
            )
      root.chmod(0o755)
      with self.assertRaisesRegex(ValueError, 'directory lstat'):
        module._exact_directory_lstat_row(  # pylint: disable=protected-access
            root, relative='.', expected=expected_directory
        )
      root.chmod(0o700)

      artifact = root / 'artifact.bin'
      artifact.write_bytes(b'bound bytes')
      artifact.chmod(0o400)
      observed = artifact.lstat()
      digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
      expected_file = {
          'mode': '0400', 'st_dev': observed.st_dev,
          'st_ino': observed.st_ino, 'st_nlink': observed.st_nlink,
          'size_bytes': observed.st_size, 'sha256': digest,
      }

      def validate_file(values):
        return module._exact_regular_lstat_binding(  # pylint: disable=protected-access
            artifact, relative='artifact.bin', **values
        )

      self.assertEqual(validate_file(expected_file)['sha256'], digest)
      for field in ('mode', 'st_dev', 'st_ino', 'st_nlink',
                    'size_bytes', 'sha256'):
        with self.subTest(file_field=field):
          changed = dict(expected_file)
          if field == 'mode':
            changed[field] = '0600'
          elif field == 'sha256':
            changed[field] = '0' * 64
          else:
            changed[field] += 1
          with self.assertRaisesRegex(ValueError, 'file (lstat row|bytes)'):
            validate_file(changed)

  def test_real_renameat2_probe_preserves_collision_orphan(self):
    with tempfile.TemporaryDirectory() as directory, self._roots(
        Path(directory)
    ) as (external, _):
      preflight._register_external_cache()  # pylint: disable=protected-access
      probe = preflight._formal_publication_probe()  # pylint: disable=protected-access
      self.assertTrue(probe['supported'])
      self.assertEqual(probe['collision_errno'], 17)
      self.assertTrue(probe['collision_no_replace_exact'])
      final = external / probe['successful_final_binding']['path']
      orphan = external / probe['collision_temp_binding']['path']
      self.assertTrue(final.is_file())
      self.assertTrue(orphan.is_file())
      self.assertEqual(final.stat().st_mode & 0o777, 0o400)
      self.assertEqual(orphan.stat().st_mode & 0o777, 0o400)

  def test_probe_live_binding_rejects_mode_link_inode_and_path_drift(self):
    with tempfile.TemporaryDirectory() as directory, self._roots(
        Path(directory)
    ) as (external, _):
      preflight._register_external_cache()  # pylint: disable=protected-access
      probe = preflight._formal_publication_probe()  # pylint: disable=protected-access
      validate = preflight.bootstrap.validate_publication_probe_live_bindings
      validate(probe)
      final = external / probe['successful_final_binding']['path']
      orphan = external / probe['collision_temp_binding']['path']

      final.chmod(0o600)
      with self.assertRaisesRegex(ValueError, 'binding changed'):
        validate(probe)
      final.chmod(0o400)

      extra_link = external / 'probe-extra-link'
      os.link(final, extra_link)
      try:
        with self.assertRaisesRegex(ValueError, 'binding changed'):
          validate(probe)
      finally:
        extra_link.unlink()

      swap = external / 'probe-swap'
      final.rename(swap)
      orphan.rename(final)
      swap.rename(orphan)
      with self.assertRaisesRegex(ValueError, 'binding changed'):
        validate(probe)
      orphan.rename(swap)
      final.rename(orphan)
      swap.rename(final)

      payload = final.read_bytes()
      replacement = external / 'probe-replacement'
      replacement.write_bytes(payload)
      replacement.chmod(0o400)
      self.assertNotEqual(replacement.stat().st_ino, final.stat().st_ino)
      displaced = external / 'probe-displaced-original'
      final.rename(displaced)
      replacement.rename(final)
      with self.assertRaisesRegex(ValueError, 'binding changed'):
        validate(probe)

      unsafe = copy.deepcopy(probe)
      unsafe['successful_final_binding']['path'] = '../escape'
      with self.assertRaisesRegex(ValueError, 'unsafe'):
        validate(unsafe)

  def test_publication_syscall_order_and_short_write_loop(self):
    with tempfile.TemporaryDirectory() as directory, self._roots(
        Path(directory)
    ):
      preflight._register_external_cache()  # pylint: disable=protected-access
      module = preflight.bootstrap
      events = []
      real_write = module.os.write
      real_fsync = module.os.fsync
      real_fchmod = module.os.fchmod
      real_rename = module._rename_noreplace  # pylint: disable=protected-access

      def short_write(fd, payload):
        events.append('write')
        return real_write(fd, bytes(payload[:1]))

      def fsync(fd):
        events.append('fsync')
        return real_fsync(fd)

      def fchmod(fd, mode):
        events.append('fchmod')
        return real_fchmod(fd, mode)

      def rename(parent_fd, temporary, final):
        events.append('renameat2')
        return real_rename(parent_fd, temporary, final)

      with (
          mock.patch.object(module.os, 'write', side_effect=short_write),
          mock.patch.object(module.os, 'fsync', side_effect=fsync),
          mock.patch.object(module.os, 'fchmod', side_effect=fchmod),
          mock.patch.object(module, '_rename_noreplace', side_effect=rename),
      ):
        result = module.publish_bytes(
            'external_cache', 'ordered.bin', b'abc', artifact_role='test'
        )
      self.assertEqual(result['publication_ordinal'], 0)
      rename_index = events.index('renameat2')
      fchmod_index = events.index('fchmod')
      self.assertGreaterEqual(events[:fchmod_index].count('fsync'), 1)
      self.assertGreaterEqual(events[fchmod_index:rename_index].count('fsync'), 1)
      self.assertGreaterEqual(events[rename_index:].count('fsync'), 1)
      self.assertEqual(events.count('renameat2'), 1)
      self.assertGreaterEqual(events.count('write'), 3)

  def test_zero_write_consumes_ordinal_and_preserves_orphan(self):
    with tempfile.TemporaryDirectory() as directory, self._roots(
        Path(directory)
    ):
      preflight._register_external_cache()  # pylint: disable=protected-access
      module = preflight.bootstrap
      with mock.patch.object(module.os, 'write', return_value=0):
        with self.assertRaises(module.PublicationError) as caught:
          module.publish_bytes(
              'external_cache', 'zero.bin', b'x', artifact_role='zero'
          )
      self.assertEqual(caught.exception.publication_failure['failure_stage'], 'write')
      audit = module.publication_audit(
          'external_cache', caught.exception.publication_failure
      )
      self.assertEqual(audit['temporary_orphan_count'], 1)
      result = module.publish_bytes(
          'external_cache', 'after.bin', b'y', artifact_role='after'
      )
      self.assertEqual(result['publication_ordinal'], 1)

  def test_all_fourteen_failure_stages_are_reached_by_real_control_flow(self):
    stages = (
        'parent_open', 'parent_validation', 'final_preexistence',
        'temp_open', 'temp_validation', 'write', 'first_file_fsync',
        'fchmod', 'second_file_fsync', 'readback', 'rename_noreplace',
        'post_rename_validation', 'parent_fsync', 'final_revalidation',
    )
    for stage in stages:
      with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
        with self._roots(Path(directory)) as (external, _):
          preflight._register_external_cache()  # pylint: disable=protected-access
          module = preflight.bootstrap
          final = f'{stage}.bin'
          payload = b'x'
          caught = None
          if stage == 'parent_open':
            old = external.with_name('external-cache-old')
            external.rename(old)
            external.mkdir(mode=0o700)
            with self.assertRaises(module.PublicationError) as caught:
              module.publish_bytes(
                  'external_cache', final, payload,
                  artifact_role='failure_stage_test'
              )
          elif stage == 'parent_validation':
            real_fstat = module.os.fstat
            calls = [0]
            def bad_parent(fd):
              observed = real_fstat(fd)
              calls[0] += 1
              if calls[0] == 1:
                values = list(observed)
                values[0] = stat.S_IFREG | 0o700
                return os.stat_result(values)
              return observed
            with mock.patch.object(module.os, 'fstat', side_effect=bad_parent):
              with self.assertRaises(module.PublicationError) as caught:
                module.publish_bytes(
                    'external_cache', final, payload,
                    artifact_role='failure_stage_test'
                )
          elif stage == 'final_preexistence':
            (external / final).write_bytes(b'old')
            with self.assertRaises(module.PublicationError) as caught:
              module.publish_bytes(
                  'external_cache', final, payload,
                  artifact_role='failure_stage_test'
              )
          elif stage == 'temp_open':
            nonce = 'a' * 32
            (external / f'.v3345.tmp.{os.getpid()}.000000.{nonce}').write_bytes(b'old')
            with mock.patch.object(module.secrets, 'token_hex', return_value=nonce):
              with self.assertRaises(module.PublicationError) as caught:
                module.publish_bytes(
                    'external_cache', final, payload,
                    artifact_role='failure_stage_test'
                )
          elif stage == 'temp_validation':
            real_fstat = module.os.fstat
            calls = [0]
            def bad_temp(fd):
              observed = real_fstat(fd)
              calls[0] += 1
              if calls[0] == 2:
                values = list(observed)
                values[0] = observed.st_mode | 0o044
                return os.stat_result(values)
              return observed
            with mock.patch.object(module.os, 'fstat', side_effect=bad_temp):
              with self.assertRaises(module.PublicationError) as caught:
                module.publish_bytes(
                    'external_cache', final, payload,
                    artifact_role='failure_stage_test'
                )
          else:
            handle = module.PublicationHandle(
                'external_cache', final, 'failure_stage_test'
            )
            if stage == 'write':
              with mock.patch.object(module.os, 'write', return_value=0):
                with self.assertRaises(module.PublicationError) as caught:
                  handle.write(payload)
            else:
              handle.write(payload)
              real_fsync = module.os.fsync
              if stage in {'first_file_fsync', 'second_file_fsync'}:
                calls = [0]
                target = 1 if stage == 'first_file_fsync' else 2
                def fail_fsync(fd):
                  calls[0] += 1
                  if calls[0] == target:
                    raise OSError(errno.EIO, 'injected fsync')
                  return real_fsync(fd)
                patcher = mock.patch.object(
                    module.os, 'fsync', side_effect=fail_fsync
                )
              elif stage == 'fchmod':
                patcher = mock.patch.object(
                    module.os, 'fchmod', side_effect=OSError(
                        errno.EIO, 'injected chmod'
                    )
                )
              elif stage == 'readback':
                patcher = mock.patch.object(
                    module, '_read_fd_bytes', side_effect=OSError(
                        errno.EIO, 'injected readback'
                    )
                )
              elif stage == 'rename_noreplace':
                patcher = mock.patch.object(
                    module, '_rename_noreplace', return_value=(-1, errno.EIO)
                )
              elif stage in {'post_rename_validation', 'final_revalidation'}:
                original_validate = handle._validate_final  # pylint: disable=protected-access
                calls = [0]
                target = 1 if stage == 'post_rename_validation' else 2
                def fail_validation(*args):
                  calls[0] += 1
                  if calls[0] == target:
                    raise OSError(errno.EIO, 'injected final validation')
                  return original_validate(*args)
                patcher = mock.patch.object(
                    handle, '_validate_final', side_effect=fail_validation
                )
              elif stage == 'parent_fsync':
                def fail_parent(fd):
                  if fd == handle.parent_fd and handle.rename_succeeded:
                    raise OSError(errno.EIO, 'injected parent fsync')
                  return real_fsync(fd)
                patcher = mock.patch.object(
                    module.os, 'fsync', side_effect=fail_parent
                )
              else:
                raise AssertionError(stage)
              with patcher:
                with self.assertRaises(module.PublicationError) as caught:
                  handle.finalize(payload)
          self.assertIsNotNone(caught)
          failure = caught.exception.publication_failure
          self.assertEqual(failure['failure_stage'], stage)
          self.assertEqual(
              set(failure), set(module.PUBLICATION_FAILURE_OBJECT_KEYS)
          )
          self.assertEqual(failure['publication_ordinal'], 0)
          self.assertEqual(module._PUBLICATION_ORDINAL, 1)  # pylint: disable=protected-access
          self.assertEqual(
              len(module._PUBLICATION_SUCCESS.get('external_cache', {})), 0  # pylint: disable=protected-access
          )

  def test_final_and_temp_special_entries_fail_without_following(self):
    kinds = ('symlink', 'fifo', 'socket', 'directory')

    def create(path: Path, kind: str):
      holder = None
      if kind == 'symlink':
        path.symlink_to('does-not-exist')
      elif kind == 'fifo':
        os.mkfifo(path)
      elif kind == 'socket':
        holder = socket.socket(socket.AF_UNIX)
        holder.bind(str(path))
      else:
        path.mkdir()
      return holder

    for location in ('final', 'temp'):
      for kind in kinds:
        with self.subTest(location=location, kind=kind), tempfile.TemporaryDirectory() as directory:
          with self._roots(Path(directory)) as (external, _):
            preflight._register_external_cache()  # pylint: disable=protected-access
            module = preflight.bootstrap
            final = external / 'blocked.bin'
            if location == 'final':
              blocked = final
            else:
              nonce = 'a' * 32
              blocked = external / (
                  f'.v3345.tmp.{os.getpid()}.000000.{nonce}'
              )
            holder = create(blocked, kind)
            try:
              nonce_patch = (
                  mock.patch.object(module.secrets, 'token_hex', return_value='a' * 32)
                  if location == 'temp' else mock.patch.object(
                      module.secrets, 'token_hex', wraps=module.secrets.token_hex
                  )
              )
              with nonce_patch, self.assertRaises(module.PublicationError) as caught:
                module.publish_bytes(
                    'external_cache', 'blocked.bin', b'x', artifact_role='special'
                )
              state_name = 'temp_state' if location == 'temp' else 'final_state'
              self.assertEqual(
                  caught.exception.publication_failure[state_name]['entry_type'],
                  kind,
              )
            finally:
              if holder is not None:
                holder.close()

  def test_registered_parent_swap_and_uncertain_final_fail_closed(self):
    with tempfile.TemporaryDirectory() as directory, self._roots(
        Path(directory)
    ) as (external, _):
      preflight._register_external_cache()  # pylint: disable=protected-access
      old = external.with_name('external-cache-old')
      external.rename(old)
      external.mkdir(mode=0o700)
      with self.assertRaises(preflight.bootstrap.PublicationError) as caught:
        preflight.bootstrap.publish_bytes(
            'external_cache', 'swap.bin', b'x', artifact_role='swap'
        )
      self.assertEqual(caught.exception.publication_failure['failure_stage'], 'parent_open')

    with tempfile.TemporaryDirectory() as directory, self._roots(
        Path(directory)
    ):
      preflight._register_external_cache()  # pylint: disable=protected-access
      module = preflight.bootstrap
      handle = module.PublicationHandle(
          'external_cache', 'uncertain.bin', 'uncertain'
      )
      handle.write(b'x')
      real_fsync = module.os.fsync

      def fsync(fd):
        if fd == handle.parent_fd and handle.rename_succeeded:
          raise OSError(errno.EIO, 'injected parent fsync failure')
        return real_fsync(fd)

      with mock.patch.object(module.os, 'fsync', side_effect=fsync):
        with self.assertRaises(module.PublicationError) as caught:
          handle.finalize(b'x')
      self.assertEqual(caught.exception.publication_failure['failure_stage'], 'parent_fsync')
      audit = module.publication_audit(
          'external_cache', caught.exception.publication_failure
      )
      self.assertEqual(audit['durability_uncertain_final_count'], 1)
      self.assertTrue(module.terminal_publication_available('external_cache'))

  def test_parent_swap_between_require_and_open_creates_no_entry(self):
    with tempfile.TemporaryDirectory() as directory, self._roots(
        Path(directory)
    ) as (external, _):
      preflight._register_external_cache()  # pylint: disable=protected-access
      module = preflight.bootstrap
      old = external.with_name('external-cache-old')
      real_open = module.os.open
      swapped = [False]

      def swap_then_open(path, flags, *args, **kwargs):
        if Path(path) == external and not swapped[0]:
          swapped[0] = True
          external.rename(old)
          external.mkdir(mode=0o700)
        return real_open(path, flags, *args, **kwargs)

      with mock.patch.object(module.os, 'open', side_effect=swap_then_open):
        with self.assertRaises(module.PublicationError) as caught:
          module.publish_bytes(
              'external_cache', 'swap-window.bin', b'x',
              artifact_role='swap_window',
          )
      self.assertEqual(
          caught.exception.publication_failure['failure_stage'],
          'parent_validation',
      )
      self.assertFalse((external / 'swap-window.bin').exists())
      self.assertFalse((old / 'swap-window.bin').exists())
      self.assertFalse(any(path.name.startswith('.v3345.tmp.') for path in external.iterdir()))
      self.assertFalse(any(path.name.startswith('.v3345.tmp.') for path in old.iterdir()))

  def test_directory_parent_swap_before_mkdir_creates_no_directory(self):
    with tempfile.TemporaryDirectory() as directory, self._roots(
        Path(directory)
    ) as (external, _):
      preflight._register_external_cache()  # pylint: disable=protected-access
      module = preflight.bootstrap
      old = external.with_name('external-cache-old')
      real_open = module.os.open
      swapped = [False]

      def swap_then_open(path, flags, *args, **kwargs):
        if Path(path) == external and not swapped[0]:
          swapped[0] = True
          external.rename(old)
          external.mkdir(mode=0o700)
        return real_open(path, flags, *args, **kwargs)

      with mock.patch.object(module.os, 'open', side_effect=swap_then_open):
        with self.assertRaisesRegex(
            RuntimeError, 'registered inode'
        ):
          module.allocate_publication_directory(
              'external_cache', 'nested'
          )
      self.assertFalse((external / 'nested').exists())
      self.assertFalse((old / 'nested').exists())

  def test_lifecycle_parent_swap_before_openat_creates_no_file(self):
    with tempfile.TemporaryDirectory() as directory, self._roots(
        Path(directory)
    ) as (external, _):
      preflight._register_external_cache()  # pylint: disable=protected-access
      module = preflight.bootstrap
      old = external.with_name('external-cache-old')
      real_open = module.os.open
      swapped = [False]

      def swap_then_open(path, flags, *args, **kwargs):
        if Path(path) == external and not swapped[0]:
          swapped[0] = True
          external.rename(old)
          external.mkdir(mode=0o700)
        return real_open(path, flags, *args, **kwargs)

      with mock.patch.object(module.os, 'open', side_effect=swap_then_open):
        with self.assertRaisesRegex(
            RuntimeError, 'registered inode'
        ):
          module.create_empty_lifecycle_file(
              'external_cache', 'allocation.lock', mode=0o600
          )
      self.assertFalse((external / 'allocation.lock').exists())
      self.assertFalse((old / 'allocation.lock').exists())

  def test_probe_happens_before_jax_import(self):
    source = Path(preflight.__file__).read_text(encoding='utf-8')
    tree = ast.parse(source)
    top_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    self.assertNotIn('run_device_preflight_v3_3', top_imports)
    self.assertLess(
        source.index('probe = _formal_publication_probe()'),
        source.index('observation = _collect_observation(pre_import)'),
    )

  def test_allocation_is_exact_0000_and_consumed(self):
    with tempfile.TemporaryDirectory() as directory, self._roots(
        Path(directory)
    ) as (_, root):
      preflight._allocate_preflight_root()  # pylint: disable=protected-access
      self.assertEqual(
          {path.name for path in root.iterdir()},
          {'.allocation.lock', '.preflight_0000.reserved'},
      )
      self.assertEqual((root / '.allocation.lock').stat().st_mode & 0o777, 0o600)
      self.assertEqual(
          (root / '.preflight_0000.reserved').stat().st_mode & 0o777, 0o400
      )
      with self.assertRaises(FileExistsError):
        preflight._allocate_preflight_root()  # pylint: disable=protected-access

  def test_full_preflight_success_has_exact_five_file_tree(self):
    prior_prefix = (
        preflight.bootstrap.validate_prior_v3_3_4_3_consumed_preflight_prefix()
    )
    prior_v3_3_4_4_prefix = (
        preflight.bootstrap.validate_prior_v3_3_4_4_consumed_preflight_prefix()
    )
    with tempfile.TemporaryDirectory() as directory:
      base = Path(directory)
      source = Path(preflight.__file__).resolve()
      relative = source.relative_to(
          preflight.bootstrap._REPO  # pylint: disable=protected-access
      ).as_posix()
      frozen = {
          'preflight_script_version': (
              preflight.bootstrap.PREFLIGHT_SCRIPT_VERSION
          ),
          'file_sha256': {
              relative: hashlib.sha256(source.read_bytes()).hexdigest(),
          },
      }
      freeze_path = base / 'freeze.json'
      freeze_path.write_text(
          json.dumps(frozen, sort_keys=True), encoding='utf-8'
      )
      freeze_digest = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
      authorization = {
          'git_head': 'a' * 40,
          'freeze_path': str(freeze_path.resolve()),
          'freeze_sha256': freeze_digest,
          'freeze_size_bytes': freeze_path.stat().st_size,
          'live_equals_git_show': True, 'tracked_clean': True,
          'authorization_source': 'external_post_commit_audit',
      }
      version_proof = preflight.bootstrap.validate_preflight_version_contract(
          frozen
      )
      freeze_record = {
          'path': str(freeze_path.resolve()), 'sha256': freeze_digest,
          'size_bytes': freeze_path.stat().st_size,
          'external_freeze_authorization': authorization,
          'preflight_version_proof': version_proof,
          'prior_v3_3_4_3_consumed_preflight_prefix': prior_prefix,
          'prior_v3_3_4_3_consumed_preflight_prefix_content_binding': (
              preflight.bootstrap.canonical_content_binding(prior_prefix)
          ),
          'prior_v3_3_4_4_consumed_preflight_prefix': prior_v3_3_4_4_prefix,
          'prior_v3_3_4_4_consumed_preflight_prefix_content_binding': dict(
              preflight.bootstrap.V3_3_4_4_CONSUMED_PREFIX_BINDING
          ),
      }

      def observation(pre_import):
        return {
            'pid': os.getpid() + 100000,
            'v3_3_4_5_runtime_environment': {
                'cache_environment': pre_import,
            },
        }

      with self._roots(base) as (_, root), mock.patch.object(
          preflight.bootstrap, 'FREEZE_PATH', freeze_path
      ), mock.patch.object(
          preflight, 'FREEZE_PATH', freeze_path
      ), mock.patch.object(
          preflight, '_validate_authorized_freeze', return_value=freeze_record
      ), mock.patch.object(
          preflight, '_collect_observation', side_effect=observation
      ), mock.patch.object(
          preflight.bootstrap, 'validate_external_freeze_authorization',
          return_value=authorization,
      ):
        record_path, passed = preflight.run_preflight()
        self.assertTrue(passed)
        self.assertEqual(
            {path.name for path in root.iterdir()},
            set(preflight.bootstrap.PREFLIGHT_CONTRACT['root_membership']),
        )
        record = json.loads(record_path.read_text(encoding='utf-8'))
        self.assertEqual(
            set(record),
            set(preflight.bootstrap.PREFLIGHT_CONTRACT['record_keys']),
        )
        self.assertEqual(record['status'], 'pass')
        self.assertTrue(record['atomic_publication_probe']['supported'])

        # The entry-time external branch must still reject this consumed root,
        # while the phase-specific parent validator accepts it without role
        # spoofing and produces the exact 12-key START subrecord.
        with self.assertRaises(FileExistsError):
          preflight.bootstrap.validate_preflight_state_for_role()
        state = (
            preflight.bootstrap.validate_completed_external_preflight_state()
        )
        self.assertEqual((state['cache_role'], state['file_count']),
                         ('external_preflight', 5))
        _, successful = launcher._validate_preflight_record(  # pylint: disable=protected-access
            record_path
        )
        self.assertEqual(len(successful), 12)
        self.assertEqual(successful['external_pid'], os.getpid() + 100000)
        self.assertEqual(
            successful[
                'prior_v3_3_4_4_consumed_preflight_prefix_content_binding'
            ],
            preflight.bootstrap.V3_3_4_4_CONSUMED_PREFIX_BINDING,
        )
        with mock.patch.object(
            preflight.bootstrap,
            'validate_completed_external_preflight_state',
            side_effect=[state, {**state, 'tree_sha256': '0' * 64}],
        ), self.assertRaisesRegex(ValueError, 'changed during'):
          launcher._validate_preflight_record(  # pylint: disable=protected-access
              record_path
          )
        routing_tampers = {
            'role': {preflight.bootstrap.CACHE_ROLE_ENVIRONMENT: 'model'},
            'root': {preflight.bootstrap.CACHE_ROOT_ENVIRONMENT: '/wrong'},
            'triton': {'TRITON_CACHE_DIR': '/wrong'},
            'xdg': {'XDG_CACHE_HOME': '/wrong'},
        }
        for label, environment in routing_tampers.items():
          with self.subTest(parent_routing=label), mock.patch.dict(
              os.environ, environment,
          ), self.assertRaisesRegex(ValueError, 'parent routing'):
            preflight.bootstrap.validate_completed_external_preflight_state()

        original_record = copy.deepcopy(record)

        def replace_record(changed):
          record_path.chmod(0o600)
          record_path.write_text(
              json.dumps(changed, indent=2, sort_keys=True, allow_nan=False),
              encoding='utf-8',
          )
          record_path.chmod(0o400)

        mutations = {}
        changed = copy.deepcopy(original_record)
        changed['unexpected'] = True
        mutations['record_key'] = changed
        changed = copy.deepcopy(original_record)
        changed['status'] = 'fail'
        mutations['status'] = changed
        changed = copy.deepcopy(original_record)
        changed['failure'] = {'type': 'Injected'}
        mutations['failure'] = changed
        changed = copy.deepcopy(original_record)
        changed['observation']['pid'] = os.getpid()
        mutations['child_pid'] = changed
        changed = copy.deepcopy(original_record)
        changed['external_cache_hit_evidence']['cache_hit'] = True
        mutations['cache_hit'] = changed
        changed = copy.deepcopy(original_record)
        changed['atomic_publication_probe']['supported'] = False
        mutations['probe'] = changed
        changed = copy.deepcopy(original_record)
        changed['freeze']['unexpected'] = True
        mutations['nested_freeze_key'] = changed
        changed = copy.deepcopy(original_record)
        changed['external_freeze_authorization']['git_head'] = 'b' * 40
        mutations['authorization'] = changed
        changed = copy.deepcopy(original_record)
        changed['external_cache_post_observation']['tree_sha256'] = '0' * 64
        mutations['cache_tree'] = changed
        for label, changed in mutations.items():
          with self.subTest(completed_record_tamper=label):
            replace_record(changed)
            with self.assertRaises((ValueError, RuntimeError)):
              preflight.bootstrap.validate_completed_external_preflight_state()
        replace_record(original_record)

        triton = preflight.bootstrap.PREFLIGHT_KERNEL_CACHE_DIR / 'triton'
        triton.chmod(0o755)
        with self.assertRaisesRegex(ValueError, 'cache-directory mode'):
          preflight.bootstrap.validate_completed_external_preflight_state()
        triton.chmod(0o700)
        root.chmod(0o755)
        with self.assertRaisesRegex(ValueError, '0700'):
          preflight.bootstrap.validate_completed_external_preflight_state()
        root.chmod(0o700)
        record_path.chmod(0o600)
        with self.assertRaisesRegex(ValueError, 'mode changed'):
          preflight.bootstrap.validate_completed_external_preflight_state()
        record_path.chmod(0o400)
        for label, create in (
            ('extra', lambda path: path.write_bytes(b'extra')),
            ('symlink', lambda path: path.symlink_to('preflight_0000.json')),
            ('fifo', lambda path: os.mkfifo(path)),
            ('directory', lambda path: path.mkdir()),
        ):
          with self.subTest(completed_tree_tamper=label):
            unsafe = root / f'unsafe-{label}'
            create(unsafe)
            try:
              with self.assertRaises((ValueError, RuntimeError)):
                preflight.bootstrap.validate_completed_external_preflight_state()
            finally:
              if unsafe.is_dir() and not unsafe.is_symlink():
                unsafe.rmdir()
              else:
                unsafe.unlink()

  def test_post_import_cache_environment_mutation_fails(self):
    with tempfile.TemporaryDirectory() as directory, self._roots(
        Path(directory)
    ):
      pre_import = preflight._register_external_cache()  # pylint: disable=protected-access
      with mock.patch.dict(os.environ, {'CUDA_CACHE_DISABLE': '0'}):
        with self.assertRaises(ValueError):
          preflight.bootstrap.assert_live_cache_environment_matches(pre_import)

  def test_launcher_order_validates_before_model_cache_and_start(self):
    source = Path(launcher.__file__).read_text(encoding='utf-8')
    tokens = (
        'path_absence = bootstrap.validate_gate_a_path_absence()',
        "run_device_preflight_v3_3_4_5.py'), '--run'",
        'preflight_record, successful_preflight = _validate_preflight_record(path)',
        'model_cache_environment = _allocate_cache(',
        "bootstrap.allocate_publication_directory('model_run')",
        "bootstrap.OUTPUT_DIR / 'ATTEMPT_STARTED.json'",
        'allow_started_output=True',
        'live_model_cache = bootstrap.cache_output_tree_binding(',
        "if cache_evidence['cache_hit']:",
        'runpy.run_path',
    )
    positions = [source.index(token) for token in tokens]
    self.assertEqual(positions, sorted(positions))

  def test_launcher_cache_gate_is_stdlib_only_before_runner_import(self):
    source = Path(launcher.__file__).read_text(encoding='utf-8')
    tree = ast.parse(source)
    top_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    self.assertFalse({'jax', 'jax.numpy', 'dna_model', 'interpretability'} & top_imports)
    cache_gate = source.index(
        'live_model_cache = bootstrap.cache_output_tree_binding('
    )
    self.assertLess(cache_gate, source.index('runpy.run_path'))
    self.assertLess(
        source.index("if cache_evidence['cache_hit']"),
        source.index('runpy.run_path'),
    )

  def test_launcher_pre_import_cache_hit_publishes_exact_four_file_terminal(self):
    with tempfile.TemporaryDirectory() as directory:
      base = Path(directory)
      output = base / 'run'
      model_cache = base / 'model-cache'
      model_cache.mkdir(mode=0o700)
      (model_cache / 'triton').mkdir(mode=0o700)
      (model_cache / 'xdg').mkdir(mode=0o700)
      module = launcher.bootstrap
      old_output = module.OUTPUT_DIR
      old_cache = module.MODEL_KERNEL_CACHE_DIR
      old_roots = dict(module.PUBLICATION_ROOTS)
      module.OUTPUT_DIR = output
      module.MODEL_KERNEL_CACHE_DIR = model_cache
      module.PUBLICATION_ROOTS['model_run'] = output
      module.PUBLICATION_ROOTS['model_cache'] = model_cache
      for value in (
          module._PUBLICATION_SUCCESS, module._PUBLICATION_TEMP_ORPHANS,
          module._PUBLICATION_UNCERTAIN_FINALS,
          module._PUBLICATION_PREEXISTING,
          module._PUBLICATION_UNBINDABLE_FAILURES,
      ):
        value.clear()
      module._PUBLICATION_DIRECTORIES.clear()
      module._PUBLICATION_ORDINAL = 0
      try:
        module.allocate_publication_directory('model_run')
        fresh = module.cache_output_tree_binding(model_cache)
        source_audit = dict(zip(
            module.SOURCE_INPUT_AUDIT_KEYS,
            (True, True, True, True, None, None, None, None), strict=True,
        ))
        start = {
            'source_input_audit': source_audit,
            'same_process_preflight': {'model_cache_pre_import': fresh},
            'external_freeze_authorization': {
                'git_head': 'a' * 40, 'freeze_path': '/fixture/freeze.json',
                'freeze_sha256': 'f' * 64, 'freeze_size_bytes': 1,
                'live_equals_git_show': True, 'tracked_clean': True,
                'authorization_source': 'external_post_commit_audit',
            },
            'freeze_sha256': 'f' * 64, 'git_head': 'a' * 40,
            'started_at_unix_s': time.time(),
            'prior_v3_3_3_binding': {},
            'prior_v3_3_3_1_archive_binding': {},
            'prior_v3_3_4_3_consumed_preflight_prefix': (
                module.validate_prior_v3_3_4_3_consumed_preflight_prefix()
            ),
        }
        start[
            'prior_v3_3_4_3_consumed_preflight_prefix_content_binding'
        ] = module.canonical_content_binding(
            start['prior_v3_3_4_3_consumed_preflight_prefix']
        )
        start['prior_v3_3_4_4_consumed_preflight_prefix'] = (
            module.validate_prior_v3_3_4_4_consumed_preflight_prefix()
        )
        start[
            'prior_v3_3_4_4_consumed_preflight_prefix_content_binding'
        ] = dict(module.V3_3_4_4_CONSUMED_PREFIX_BINDING)
        launcher._publish_new(output / 'ATTEMPT_STARTED.json', start)  # pylint: disable=protected-access
        (model_cache / 'adverse.bin').write_bytes(b'cache input')
        live = module.cache_output_tree_binding(model_cache)
        evidence = launcher._model_cache_hit_evidence(live)  # pylint: disable=protected-access
        self.assertTrue(evidence['cache_hit'])
        launcher._publish_model_cache_pre_import_stop(  # pylint: disable=protected-access
            start=start, live_cache=live, evidence=evidence,
            error=RuntimeError('cache hit'),
        )
        self.assertEqual(
            {path.name for path in output.iterdir()},
            {'ATTEMPT_STARTED.json', 'MODEL_CACHE_PRE_IMPORT_HIT.json',
             'RAW_MANIFEST.json', 'RUN_COMPLETE.json'},
        )
        completion = json.loads((output / 'RUN_COMPLETE.json').read_text())
        self.assertEqual(
            set(completion), set(module.TERMINAL_CONTRACT['run_complete_keys'])
        )
        self.assertEqual(completion['status'], 'controlled_stop_cache_hit')
        self.assertEqual(completion['model_apply_attempt_count'], 0)
      finally:
        module.OUTPUT_DIR = old_output
        module.MODEL_KERNEL_CACHE_DIR = old_cache
        module.PUBLICATION_ROOTS.clear()
        module.PUBLICATION_ROOTS.update(old_roots)
        module._PUBLICATION_DIRECTORIES.clear()

  def test_launcher_dry_cache_is_nonproduction_and_removed(self):
    root = launcher._temporary_gate_cache()  # pylint: disable=protected-access
    try:
      self.assertTrue(root.name.startswith('alphagenome-v3.3.4.5-dry-cache.'))
      self.assertTrue((root / 'triton').is_dir())
      self.assertTrue((root / 'xdg').is_dir())
    finally:
      launcher._remove_empty_gate_cache(root)  # pylint: disable=protected-access
    self.assertFalse(root.exists())

  def test_revised_serializers_all_bind_consumed_prefix_and_content(self):
    module = preflight.bootstrap
    required = {
        'prior_v3_3_4_3_consumed_preflight_prefix',
        'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
        'prior_v3_3_4_4_consumed_preflight_prefix',
        'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
    }
    keysets = (
        module.PREFLIGHT_CONTRACT['record_keys'],
        module.START_RECORD_KEYS,
        module.POST_START_PROVENANCE_FAILURE_KEYS,
        module.TERMINAL_CONTRACT['run_complete_keys'],
        module.PUBLICATION_TERMINAL_FAILURE_KEYS,
        module.NONPUBLICATION_TERMINAL_CONTRACT_V3_3_4_5['keys'],
    )
    self.assertEqual([len(value) for value in keysets], [22, 38, 24, 68, 35, 62])
    for keyset in keysets:
      self.assertTrue(required <= set(keyset))

  def test_freeze_generator_is_deterministic_exact_86_132_132(self):
    generator = importlib.import_module(
        'generate_encoder_skip_ood_sidecar_v3_3_4_5_freeze'
    )
    first = generator.build_freeze()
    second = generator.build_freeze()
    first_bytes = json.dumps(
        first, indent=2, sort_keys=True, allow_nan=False
    ).encode('utf-8')
    second_bytes = json.dumps(
        second, indent=2, sort_keys=True, allow_nan=False
    ).encode('utf-8')
    self.assertEqual(first_bytes, second_bytes)
    self.assertEqual(
        (len(first), len(first['file_sha256']),
         len(first['source_inventory_contract']['rows'])),
        (86, 132, 132),
    )
    self.assertIn('nonpublication_terminal_contract_v3_3_4_5', first)
    self.assertNotIn('nonpublication_terminal_contract_v3_3_4_4', first)
    predecessor = json.loads(
        preflight.bootstrap.V3_3_4_4_FREEZE_PATH.read_text(encoding='utf-8')
    )
    added = set(first['file_sha256']) - set(predecessor['file_sha256'])
    self.assertEqual(added, set(generator._EXTRA_FILES))  # pylint: disable=protected-access
    rows = {
        row['path']: row for row in first['source_inventory_contract']['rows']
    }
    self.assertEqual(set(rows), set(first['file_sha256']))
    for relative, digest in first['file_sha256'].items():
      path = generator._REPO / relative  # pylint: disable=protected-access
      self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
      self.assertEqual(rows[relative]['sha256'], digest)
      self.assertEqual(rows[relative]['size_bytes'], path.stat().st_size)
      expected_mode = '100755' if relative.endswith('.sh') else '100644'
      if relative in added:
        self.assertEqual(rows[relative]['git_mode'], expected_mode)
        self.assertEqual(
            stat.S_IMODE(path.stat().st_mode),
            0o755 if relative.endswith('.sh') else 0o644,
        )
    prefix = first['prior_v3_3_4_3_consumed_preflight_prefix']
    self.assertEqual(
        preflight.bootstrap.canonical_content_binding(prefix),
        {'sha256': 'c42ce8bd47918daf90affba701eb1dc193c1ba2cbb38e3b0f836e8b81f306f88',
         'size_bytes': 6418},
    )
    v3_3_4_4_prefix = first[
        'prior_v3_3_4_4_consumed_preflight_prefix'
    ]
    self.assertEqual(
        preflight.bootstrap.canonical_content_binding(v3_3_4_4_prefix),
        {'sha256': 'efcb6d8946666d104d7458c0f13cc8f53e6dfaa1a30a2e83744f48641978f3c7',
         'size_bytes': 8653},
    )
    with tempfile.TemporaryDirectory() as directory:
      candidate = Path(directory) / 'freeze.json'
      with mock.patch.object(
          generator.bootstrap, 'FREEZE_PATH', candidate
      ), mock.patch.object(
          generator, 'build_freeze', return_value=first
      ), mock.patch('builtins.print'):
        generator.main()
      self.assertEqual(stat.S_IMODE(candidate.stat().st_mode), 0o644)
      self.assertEqual(
          candidate.read_bytes(),
          (json.dumps(
              first, indent=2, sort_keys=True, allow_nan=False
          ) + '\n').encode('utf-8'),
      )

  def test_source_contains_no_forbidden_publication_or_science(self):
    paths = [
        _HERE / name for name in (
            'analyze_encoder_skip_ood_sidecar_v3_3_4_5.py',
            'analyze_encoder_skip_ood_sidecar_v3_3_4_5.sh',
            'analyze_encoder_skip_ood_sidecar_v3_3_4_5_test.py',
            'generate_encoder_skip_ood_sidecar_v3_3_4_5_freeze.py',
            'launch_encoder_skip_ood_sidecar_v3_3_4_5.py',
            'run_device_preflight_v3_3_4_5.py',
            'run_device_preflight_v3_3_4_5_test.py',
            'run_encoder_skip_ood_sidecar_v3_3_4_5.py',
            'run_encoder_skip_ood_sidecar_v3_3_4_5.sh',
            'run_encoder_skip_ood_sidecar_v3_3_4_5_test.py',
            'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_5.py',
        )
    ]
    paths.append(
        _HERE / 'v3_wider_mechanism' /
        'encoder_skip_ood_sidecar_preflight_phase_amendment_v3_3_4_5.md'
    )
    forbidden_values = (
        'O_TMP' + 'FILE', '/proc/self/' + 'fd', 'link' + 'at',
        'os.' + 'rename', 'os.' + 'replace',
    )
    for path in paths:
      source = path.read_text(encoding='utf-8')
      if path == Path(preflight.__file__):
        for forbidden in (
            'from alphagenome_research.model', 'dna_model',
            'jax.jit', 'jax.numpy',
        ):
          self.assertNotIn(forbidden, source)
      # The historical amendment discloses old primitives in prose; runtime
      # source and tests must not contain them.
      if path.suffix != '.md':
        for forbidden in forbidden_values:
          self.assertNotIn(forbidden, source)


if __name__ == '__main__':
  unittest.main()
