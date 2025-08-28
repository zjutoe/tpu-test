#!/usr/bin/env python3
"""
TPU Monitor - A htop-like tool for TPU monitoring
Similar to nvidia-smi but for Google Cloud TPU
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime


def get_system_stats():
    """Get basic system statistics"""
    try:
        # CPU usage
        cpu_cmd = "top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1"
        cpu_usage = subprocess.check_output(cpu_cmd, shell=True, text=True).strip()
        
        # Memory usage
        mem_cmd = "free -h | grep '^Mem:' | awk '{print $3 \"/\" $2}'"
        memory_usage = subprocess.check_output(mem_cmd, shell=True, text=True).strip()
        
        # Load average
        load_cmd = "uptime | awk -F'load average:' '{print $2}'"
        load_avg = subprocess.check_output(load_cmd, shell=True, text=True).strip()
        
        return {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'load_average': load_avg.strip()
        }
    except Exception as e:
        return {
            'cpu_usage': 'N/A',
            'memory_usage': 'N/A', 
            'load_average': 'N/A'
        }


def get_tpu_processes():
    """Get Python processes that might be using TPU/XLA"""
    try:
        cmd = "ps aux | grep python | grep -v grep"
        result = subprocess.check_output(cmd, shell=True, text=True)
        
        processes = []
        for line in result.strip().split('\n'):
            if line:
                parts = line.split()
                if len(parts) >= 11:
                    processes.append({
                        'pid': parts[1],
                        'cpu': parts[2],
                        'mem': parts[3],
                        'command': ' '.join(parts[10:])[:80] + '...' if len(' '.join(parts[10:])) > 80 else ' '.join(parts[10:])
                    })
        return processes
    except Exception:
        return []


def get_xla_metrics():
    """Try to get XLA metrics if available"""
    try:
        # This only works if XLA is currently running
        result = subprocess.run([
            sys.executable, '-c', '''
import os
os.environ["PJRT_DEVICE"] = "CPU"
import torch_xla.debug.metrics as met
import json
try:
    metrics = met.short_metrics_report()
    print(json.dumps({"status": "active", "metrics": metrics}))
except Exception as e:
    print(json.dumps({"status": "inactive", "error": str(e)}))
'''
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            return {"status": "error", "error": result.stderr}
            
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}


def get_docker_containers():
    """Check for TPU-related Docker containers"""
    try:
        cmd = "docker ps --format 'table {{.Names}}\\t{{.Status}}\\t{{.Image}}' | grep -E '(tpu|xla)' || echo 'No TPU containers'"
        result = subprocess.check_output(cmd, shell=True, text=True)
        return result.strip()
    except Exception:
        return "Docker not accessible"


def display_dashboard():
    """Display TPU monitoring dashboard"""
    os.system('clear')
    
    print("🚀 TPU Monitor - Google Cloud TPU Monitoring Dashboard")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # System Stats
    print("📊 SYSTEM STATS")
    print("-" * 40)
    stats = get_system_stats()
    print(f"CPU Usage:     {stats['cpu_usage']}%")
    print(f"Memory Usage:  {stats['memory_usage']}")
    print(f"Load Average:  {stats['load_average']}")
    print()
    
    # TPU/XLA Status
    print("🧠 TPU/XLA STATUS")
    print("-" * 40)
    xla_status = get_xla_metrics()
    print(f"XLA Status: {xla_status['status'].upper()}")
    if xla_status['status'] == 'active':
        print(f"Metrics: {xla_status.get('metrics', 'N/A')}")
    elif 'error' in xla_status:
        print(f"Error: {xla_status['error'][:60]}...")
    print()
    
    # TPU Runtime Service
    print("🔧 TPU RUNTIME SERVICE")
    print("-" * 40)
    try:
        result = subprocess.check_output(
            "systemctl is-active tpu-runtime 2>/dev/null || echo 'inactive'", 
            shell=True, text=True
        ).strip()
        print(f"TPU Runtime: {result.upper()}")
        
        if result == 'active':
            # Get runtime info
            runtime_info = subprocess.check_output(
                "systemctl status tpu-runtime --no-pager -l | grep 'Active:' | cut -d':' -f2-",
                shell=True, text=True
            ).strip()
            print(f"Details: {runtime_info}")
    except Exception as e:
        print(f"TPU Runtime: UNKNOWN ({str(e)[:30]}...)")
    print()
    
    # Python Processes
    print("🐍 PYTHON PROCESSES")
    print("-" * 40)
    processes = get_tpu_processes()
    if processes:
        print(f"{'PID':<8} {'CPU%':<6} {'MEM%':<6} {'COMMAND'}")
        print("-" * 80)
        for proc in processes[:5]:  # Show top 5
            print(f"{proc['pid']:<8} {proc['cpu']:<6} {proc['mem']:<6} {proc['command']}")
    else:
        print("No Python processes found")
    print()
    
    # Docker Containers
    print("🐳 TPU CONTAINERS")
    print("-" * 40)
    containers = get_docker_containers()
    print(containers)
    print()
    
    # Environment Variables
    print("🌍 TPU ENVIRONMENT")
    print("-" * 40)
    env_vars = ['PJRT_DEVICE', 'TPU_NUM_DEVICES', 'XLA_USE_BF16', 'TPU_NAME']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var:<15}: {value}")
    
    print("\n" + "=" * 80)
    print("Press Ctrl+C to exit | Refreshing every 5 seconds...")


def main():
    """Main monitoring loop"""
    try:
        while True:
            display_dashboard()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\n👋 TPU Monitor stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()