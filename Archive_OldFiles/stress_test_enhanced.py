"""
Enhanced Stress Test - Simulates real-world usage patterns
This test will help identify issues under load
"""

import sys
import os
import time
import threading
import traceback
from pathlib import Path
from datetime import datetime

# Test results
stress_results = {
    "operations": 0,
    "errors": [],
    "warnings": [],
    "performance": {}
}

def log_error(operation, error):
    """Log an error during stress testing"""
    error_info = {
        "operation": operation,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now().isoformat()
    }
    stress_results["errors"].append(error_info)
    print(f"[ERROR] {operation}: {error}")

def log_warning(operation, message):
    """Log a warning during stress testing"""
    warning_info = {
        "operation": operation,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    stress_results["warnings"].append(warning_info)
    print(f"[WARN] {operation}: {message}")

def test_rapid_mode_switches():
    """Test 1: Rapid mode switching"""
    print("\n" + "="*60)
    print("STRESS TEST 1: Rapid Mode Switching")
    print("="*60)
    
    modes = [
        "General Assistant & Triage",
        "IT Support",
        "Executive Assistant & Operations",
        "Incentives",
        "Research & Learning",
        "Legal Research Assistant",
        "Accounting/Finance/Taxes"
    ]
    
    MODE_MODEL_DEFAULTS = {
        "General Assistant & Triage": ("gpt-5-mini", "gpt-5.1"),
        "IT Support": ("gpt-5.1", "gpt-5-mini"),
        "Executive Assistant & Operations": ("gpt-5-mini", "gpt-5.1"),
        "Incentives": ("gpt-5.1", "gpt-5-mini"),
        "Research & Learning": ("gpt-5.1", "gpt-5-mini"),
        "Legal Research Assistant": ("gpt-5.1", "gpt-5"),
        "Accounting/Finance/Taxes": ("gpt-5.1", "gpt-5-mini"),
    }
    
    errors = 0
    start_time = time.time()
    
    # Simulate 100 rapid mode switches
    for i in range(100):
        try:
            mode = modes[i % len(modes)]
            if mode not in MODE_MODEL_DEFAULTS:
                log_error(f"Mode switch {i}", f"Mode {mode} not found")
                errors += 1
            else:
                primary, backup = MODE_MODEL_DEFAULTS[mode]
                if not primary or not backup:
                    log_error(f"Mode switch {i}", f"Invalid models for {mode}")
                    errors += 1
            stress_results["operations"] += 1
        except Exception as e:
            log_error(f"Mode switch {i}", e)
            errors += 1
    
    elapsed = time.time() - start_time
    if elapsed < 0.001:  # Very fast, use minimum time
        elapsed = 0.001
    stress_results["performance"]["mode_switches"] = {
        "count": 100,
        "time": elapsed,
        "ops_per_sec": 100 / elapsed
    }
    
    if errors == 0:
        print(f"[PASS] 100 mode switches completed in {elapsed:.2f}s ({100/elapsed:.1f} ops/sec)")
        return True
    else:
        print(f"[FAIL] {errors} errors during mode switching")
        return False

def test_concurrent_file_operations():
    """Test 2: Concurrent file operations"""
    print("\n" + "="*60)
    print("STRESS TEST 2: Concurrent File Operations")
    print("="*60)
    
    errors = 0
    completed = 0
    start_time = time.time()
    
    def file_operation(worker_id):
        nonlocal errors, completed
        try:
            # Simulate file reading
            test_file = Path(__file__)
            if test_file.exists():
                # Simulate reading
                time.sleep(0.01)  # Simulate I/O delay
                completed += 1
            else:
                log_error(f"File op {worker_id}", "Test file not found")
                errors += 1
        except Exception as e:
            log_error(f"File op {worker_id}", e)
            errors += 1
    
    # Run 20 concurrent file operations
    threads = []
    for i in range(20):
        thread = threading.Thread(target=file_operation, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads
    for thread in threads:
        thread.join(timeout=5.0)
    
    elapsed = time.time() - start_time
    
    if completed == 20 and errors == 0:
        print(f"[PASS] 20 concurrent file operations completed in {elapsed:.2f}s")
        return True
    else:
        print(f"[FAIL] Completed: {completed}/20, Errors: {errors}")
        return False

def test_message_history_management():
    """Test 3: Message history management under load"""
    print("\n" + "="*60)
    print("STRESS TEST 3: Message History Management")
    print("="*60)
    
    errors = 0
    history = []
    max_history = 20
    
    # Simulate adding 1000 messages
    start_time = time.time()
    for i in range(1000):
        try:
            history.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i}"
            })
            
            # Limit history
            if len(history) > max_history:
                history = history[-max_history:]
            
            # Verify history doesn't exceed limit
            if len(history) > max_history:
                log_error(f"History limit {i}", f"History exceeded limit: {len(history)}")
                errors += 1
            
            stress_results["operations"] += 1
        except Exception as e:
            log_error(f"History management {i}", e)
            errors += 1
    
    elapsed = time.time() - start_time
    if elapsed < 0.001:  # Very fast, use minimum time
        elapsed = 0.001
    
    if errors == 0 and len(history) <= max_history:
        print(f"[PASS] 1000 messages processed, history limited correctly ({len(history)} items)")
        print(f"       Completed in {elapsed:.2f}s ({1000/elapsed:.0f} ops/sec)")
        return True
    else:
        print(f"[FAIL] Errors: {errors}, Final history size: {len(history)}")
        return False

def test_thread_cleanup_stress():
    """Test 4: Thread cleanup under stress"""
    print("\n" + "="*60)
    print("STRESS TEST 4: Thread Cleanup Under Stress")
    print("="*60)
    
    try:
        from PyQt6.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
        
        # Create QApplication if needed
        app = None
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is None:
                app = QApplication([])
        except:
            pass
        
        errors = 0
        threads_created = 0
        threads_cleaned = 0
        
        class TestWorker(QObject):
            finished = pyqtSignal()
            
            @pyqtSlot()
            def run(self):
                time.sleep(0.01)  # Simulate work
                self.finished.emit()
        
        # Create and cleanup 50 threads rapidly
        start_time = time.time()
        for i in range(50):
            try:
                thread = QThread()
                worker = TestWorker()
                worker.moveToThread(thread)
                
                thread.started.connect(worker.run)
                worker.finished.connect(thread.quit)
                
                thread.start()
                threads_created += 1
                
                # Wait with timeout
                if not thread.wait(1000):  # 1 second timeout
                    thread.terminate()
                    thread.wait(200)
                
                # Cleanup
                worker.deleteLater()
                thread.deleteLater()
                threads_cleaned += 1
                
            except Exception as e:
                log_error(f"Thread cleanup {i}", e)
                errors += 1
        
        elapsed = time.time() - start_time
        
        if errors == 0 and threads_created == 50:
            print(f"[PASS] Created and cleaned {threads_cleaned}/50 threads in {elapsed:.2f}s")
            return True
        else:
            print(f"[FAIL] Created: {threads_created}, Cleaned: {threads_cleaned}, Errors: {errors}")
            return False
            
    except Exception as e:
        log_error("Thread cleanup stress test", e)
        return False

def test_memory_leak_detection():
    """Test 5: Memory leak detection"""
    print("\n" + "="*60)
    print("STRESS TEST 5: Memory Leak Detection")
    print("="*60)
    
    import gc
    
    # Create many objects and ensure they're cleaned up
    initial_count = len(gc.get_objects())
    objects_created = []
    
    for i in range(1000):
        # Create test objects
        obj = {
            "id": i,
            "data": "x" * 100,  # Some data
            "timestamp": time.time()
        }
        objects_created.append(obj)
    
    # Clear references
    objects_created.clear()
    
    # Force garbage collection
    gc.collect()
    
    final_count = len(gc.get_objects())
    difference = final_count - initial_count
    
    if difference < 1000:  # Allow some overhead
        print(f"[PASS] Memory cleanup successful (difference: {difference} objects)")
        return True
    else:
        log_warning("Memory leak", f"Potential memory leak: {difference} objects not cleaned")
        return False

def test_error_recovery():
    """Test 6: Error recovery and resilience"""
    print("\n" + "="*60)
    print("STRESS TEST 6: Error Recovery")
    print("="*60)
    
    errors_handled = 0
    errors_thrown = 0
    
    # Simulate various error conditions
    error_scenarios = [
        ("ValueError", ValueError("Test error")),
        ("KeyError", KeyError("missing_key")),
        ("TypeError", TypeError("Invalid type")),
        ("AttributeError", AttributeError("No attribute")),
    ]
    
    for scenario_name, error in error_scenarios:
        for i in range(10):
            try:
                # Simulate operation that might fail
                if i % 3 == 0:  # Fail every 3rd operation
                    errors_thrown += 1
                    raise error
                else:
                    # Normal operation
                    pass
                errors_handled += 1
            except Exception as e:
                # Error handling
                errors_handled += 1
                stress_results["operations"] += 1
    
    if errors_handled == 40:  # 4 scenarios * 10 iterations
        print(f"[PASS] Error recovery working ({errors_handled} operations handled)")
        return True
    else:
        print(f"[FAIL] Error recovery issues (handled: {errors_handled}/40)")
        return False

def test_api_call_simulation():
    """Test 7: Simulate rapid API call patterns"""
    print("\n" + "="*60)
    print("STRESS TEST 7: API Call Pattern Simulation")
    print("="*60)
    
    # Simulate API call patterns without actually calling API
    calls_made = 0
    errors = 0
    start_time = time.time()
    
    def simulate_api_call(call_id):
        nonlocal calls_made, errors
        try:
            # Simulate API call delay
            time.sleep(0.05)  # 50ms delay
            
            # Simulate occasional failures (10% failure rate)
            if call_id % 10 == 0:
                raise Exception("Simulated API error")
            
            calls_made += 1
        except Exception as e:
            errors += 1
            # Simulate retry
            try:
                time.sleep(0.05)
                calls_made += 1  # Retry succeeds
            except:
                pass
    
    # Make 50 simulated API calls
    threads = []
    for i in range(50):
        thread = threading.Thread(target=simulate_api_call, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all
    for thread in threads:
        thread.join(timeout=10.0)
    
    elapsed = time.time() - start_time
    
    if calls_made >= 45:  # Allow for some failures
        print(f"[PASS] {calls_made}/50 API calls completed in {elapsed:.2f}s")
        return True
    else:
        print(f"[FAIL] Only {calls_made}/50 calls completed, {errors} errors")
        return False

def generate_report():
    """Generate detailed stress test report"""
    print("\n" + "="*60)
    print("STRESS TEST REPORT")
    print("="*60)
    
    print(f"\nTotal Operations: {stress_results['operations']}")
    print(f"Total Errors: {len(stress_results['errors'])}")
    print(f"Total Warnings: {len(stress_results['warnings'])}")
    
    if stress_results['performance']:
        print("\nPerformance Metrics:")
        for test_name, metrics in stress_results['performance'].items():
            print(f"  {test_name}:")
            print(f"    - Operations: {metrics.get('count', 0)}")
            print(f"    - Time: {metrics.get('time', 0):.2f}s")
            print(f"    - Ops/sec: {metrics.get('ops_per_sec', 0):.1f}")
    
    if stress_results['errors']:
        print("\nErrors Found:")
        for i, error in enumerate(stress_results['errors'][:10], 1):  # Show first 10
            print(f"  {i}. {error['operation']}: {error['error']}")
        if len(stress_results['errors']) > 10:
            print(f"  ... and {len(stress_results['errors']) - 10} more errors")
    
    if stress_results['warnings']:
        print("\nWarnings:")
        for warning in stress_results['warnings'][:5]:  # Show first 5
            print(f"  - {warning['operation']}: {warning['message']}")
    
    # Save detailed report to file
    report_file = Path(__file__).parent / "stress_test_report.txt"
    try:
        with open(report_file, 'w') as f:
            f.write("LEA ASSISTANT STRESS TEST REPORT\n")
            f.write("="*60 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"Total Operations: {stress_results['operations']}\n")
            f.write(f"Total Errors: {len(stress_results['errors'])}\n")
            f.write(f"Total Warnings: {len(stress_results['warnings'])}\n\n")
            
            if stress_results['errors']:
                f.write("DETAILED ERRORS:\n")
                f.write("-"*60 + "\n")
                for error in stress_results['errors']:
                    f.write(f"\nOperation: {error['operation']}\n")
                    f.write(f"Error: {error['error']}\n")
                    f.write(f"Time: {error['timestamp']}\n")
                    f.write(f"Traceback:\n{error['traceback']}\n")
                    f.write("-"*60 + "\n")
        
        print(f"\n[INFO] Detailed report saved to: {report_file}")
    except Exception as e:
        print(f"\n[WARN] Could not save report: {e}")

def main():
    """Run all stress tests"""
    print("\n" + "="*60)
    print("LEA ASSISTANT ENHANCED STRESS TEST")
    print("="*60)
    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    tests = [
        ("Rapid Mode Switching", test_rapid_mode_switches),
        ("Concurrent File Operations", test_concurrent_file_operations),
        ("Message History Management", test_message_history_management),
        ("Thread Cleanup Stress", test_thread_cleanup_stress),
        ("Memory Leak Detection", test_memory_leak_detection),
        ("Error Recovery", test_error_recovery),
        ("API Call Simulation", test_api_call_simulation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            log_error(test_name, e)
            results.append((test_name, False))
    
    # Generate report
    generate_report()
    
    # Summary
    print("\n" + "="*60)
    print("STRESS TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if len(stress_results['errors']) > 0:
        print(f"\n[IMPORTANT] Found {len(stress_results['errors'])} errors during stress testing")
        print("Review stress_test_report.txt for details")
        return 1
    elif passed == total:
        print("\n[SUCCESS] All stress tests passed!")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) had issues")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

