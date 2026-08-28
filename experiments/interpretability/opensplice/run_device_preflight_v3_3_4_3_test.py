#!/usr/bin/env python3
"""CPU-only contract tests for the v3.3.4.3 external preflight."""

from __future__ import annotations

from contextlib import contextmanager
import ast
import copy
import errno
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
import launch_encoder_skip_ood_sidecar_v3_3_4_3 as launcher  # pylint: disable=g-import-not-at-top
import run_device_preflight_v3_3_4_3 as preflight  # pylint: disable=g-import-not-at-top


def _environment(root: Path, role: str = 'external_preflight') -> dict[str, str]:
  return {
      'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
      'JAX_ENABLE_COMPILATION_CACHE': 'false',
      'CUDA_CACHE_DISABLE': '1',
      'ALPHAGENOME_V3_3_4_3_CACHE_ROLE': role,
      'ALPHAGENOME_V3_3_4_3_CACHE_ROOT': str(root.resolve()),
      'TRITON_CACHE_DIR': str((root / 'triton').resolve()),
      'XDG_CACHE_HOME': str((root / 'xdg').resolve()),
  }


class DevicePreflightV3343Test(unittest.TestCase):

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
            (external / f'.v3343.tmp.{os.getpid()}.000000.{nonce}').write_bytes(b'old')
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
                  f'.v3343.tmp.{os.getpid()}.000000.{nonce}'
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
      self.assertFalse(any(path.name.startswith('.v3343.tmp.') for path in external.iterdir()))
      self.assertFalse(any(path.name.startswith('.v3343.tmp.') for path in old.iterdir()))

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
    authorization = {
        'git_head': 'a' * 40, 'freeze_path': '/fixture/freeze.json',
        'freeze_sha256': 'f' * 64, 'freeze_size_bytes': 1,
        'live_equals_git_show': True, 'tracked_clean': True,
        'authorization_source': 'external_post_commit_audit',
    }
    with tempfile.TemporaryDirectory() as directory, self._roots(
        Path(directory)
    ) as (_, root), mock.patch.object(
        preflight, '_validate_authorized_freeze', return_value={
            'path': '/fixture/freeze.json', 'sha256': 'f' * 64,
            'size_bytes': 1, 'external_freeze_authorization': authorization,
        }
    ), mock.patch.object(
        preflight, '_collect_observation', return_value={
            'pid': os.getpid(),
            'v3_3_4_3_runtime_environment': {
                'cache_environment': {'cache_role': 'external_preflight'}
            },
        }
    ):
      record_path, passed = preflight.run_preflight()
      self.assertTrue(passed)
      self.assertEqual(
          {path.name for path in root.iterdir()},
          set(preflight.bootstrap.PREFLIGHT_CONTRACT['root_membership']),
      )
      record = json.loads(record_path.read_text(encoding='utf-8'))
      self.assertEqual(
          set(record), set(preflight.bootstrap.PREFLIGHT_CONTRACT['record_keys'])
      )
      self.assertEqual(record['status'], 'pass')
      self.assertTrue(record['atomic_publication_probe']['supported'])

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
        "run_device_preflight_v3_3_4_3.py'), '--run'",
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
        }
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
      self.assertTrue(root.name.startswith('alphagenome-v3.3.4.3-dry-cache.'))
      self.assertTrue((root / 'triton').is_dir())
      self.assertTrue((root / 'xdg').is_dir())
    finally:
      launcher._remove_empty_gate_cache(root)  # pylint: disable=protected-access
    self.assertFalse(root.exists())

  def test_source_contains_no_forbidden_publication_or_science(self):
    paths = [
        _HERE / name for name in (
            'analyze_encoder_skip_ood_sidecar_v3_3_4_3.py',
            'analyze_encoder_skip_ood_sidecar_v3_3_4_3.sh',
            'analyze_encoder_skip_ood_sidecar_v3_3_4_3_test.py',
            'generate_encoder_skip_ood_sidecar_v3_3_4_3_freeze.py',
            'launch_encoder_skip_ood_sidecar_v3_3_4_3.py',
            'run_device_preflight_v3_3_4_3.py',
            'run_device_preflight_v3_3_4_3_test.py',
            'run_encoder_skip_ood_sidecar_v3_3_4_3.py',
            'run_encoder_skip_ood_sidecar_v3_3_4_3.sh',
            'run_encoder_skip_ood_sidecar_v3_3_4_3_test.py',
            'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_3.py',
        )
    ]
    paths.append(
        _HERE / 'v3_wider_mechanism' /
        'encoder_skip_ood_sidecar_stage_semantics_amendment_v3_3_4_3.md'
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
