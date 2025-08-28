#!/usr/bin/env python3
"""
Fixed TPU Monitor - Avoids XLA initialization conflicts
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


def check_xla_activity():
    """Check for XLA activity without importing torch_xla"""
    try:
        # Look for XLA/TPU-related processes
        xla_processes = []
        processes = get_tpu_processes()
        
        for proc in processes:
            if any(keyword in proc['command'].lower() for keyword in ['xla', 'tpu', 'torch_xla']):
                xla_processes.append(proc)
        
        if xla_processes:
            total_cpu = sum(float(p['cpu']) for p in xla_processes if p['cpu'].replace('.', '').isdigit())
            return {
                'status': 'active',
                'processes': len(xla_processes),
                'total_cpu': f"{total_cpu:.1f}%",
                'details': f"{len(xla_processes)} XLA processes running"
            }
        else:
            return {
                'status': 'inactive',
                'processes': 0,
                'total_cpu': '0.0%',
                'details': 'No XLA processes detected'
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'details': 'Could not check XLA processes'
        }


def get_tpu_service_status():
    """Get TPU service status"""
    try:
        # Check TPU runtime service
        status_cmd = "systemctl is-active tpu-runtime 2>/dev/null || echo 'inactive'"
        status = subprocess.check_output(status_cmd, shell=True, text=True).strip()
        
        if status == 'active':
            # Get uptime
            uptime_cmd = "systemctl show tpu-runtime --property=ActiveEnterTimestamp --no-pager | cut -d'=' -f2"
            uptime_result = subprocess.check_output(uptime_cmd, shell=True, text=True).strip()
            
            return {
                'status': status.upper(),
                'uptime': uptime_result if uptime_result else 'Unknown',
                'details': 'TPU runtime service is running'
            }
        else:
            return {
                'status': status.upper(),
                'uptime': 'N/A',
                'details': 'TPU runtime service not active'
            }
    except Exception as e:
        return {
            'status': 'UNKNOWN',
            'uptime': 'N/A', 
            'details': f'Error: {str(e)[:50]}...'
        }


def display_dashboard():
    """Display TPU monitoring dashboard"""
    os.system('clear')
    
    print("🚀 TPU Monitor - Google Cloud TPU Monitoring Dashboard (Fixed)")
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
    
    # XLA Activity (without importing torch_xla)
    print("🧠 XLA ACTIVITY STATUS")
    print("-" * 40)
    xla_status = check_xla_activity()
    print(f"XLA Status:    {xla_status['status'].upper()}")
    print(f"XLA Processes: {xla_status.get('processes', 0)}")
    print(f"XLA CPU Usage: {xla_status.get('total_cpu', '0.0%')}")
    print(f"Details:       {xla_status.get('details', 'No details')}")
    print()
    
    # TPU Runtime Service
    print("🔧 TPU RUNTIME SERVICE")
    print("-" * 40)
    service_info = get_tpu_service_status()
    print(f"Service Status: {service_info['status']}")
    print(f"Service Uptime: {service_info['uptime']}")
    print(f"Details:        {service_info['details']}")
    print()
    
    # Python Processes (top 8)
    print("🐍 PYTHON PROCESSES")
    print("-" * 40)
    processes = get_tpu_processes()
    if processes:
        print(f"{'PID':<8} {'CPU%':<6} {'MEM%':<6} {'COMMAND'}")
        print("-" * 80)
        for proc in processes[:8]:  # Show top 8
            print(f"{proc['pid']:<8} {proc['cpu']:<6} {proc['mem']:<6} {proc['command']}")
    else:
        print("No Python processes found")
    print()
    
    # Environment Variables
    print("🌍 TPU ENVIRONMENT")
    print("-" * 40)
    env_vars = ['PJRT_DEVICE', 'TPU_NUM_DEVICES', 'XLA_USE_BF16', 'TPU_NAME']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var:<15}: {value}")
    
    # Training Files Status
    print()
    print("📁 TRAINING FILES")
    print("-" * 40)
    try:
        model_files = subprocess.check_output("ls -la *.pth 2>/dev/null || echo 'No model files'", 
                                            shell=True, text=True, cwd="/home/yeminjiao/src/tpu-test").strip()
        print(model_files)
    except Exception:
        print("Could not check model files")
    
    print("\n" + "=" * 80)
    print("Press Ctrl+C to exit | Refreshing every 3 seconds...")
    print("Note: This monitor avoids XLA import conflicts")


def main():
    """Main monitoring loop"""
    try:
        while True:
            display_dashboard()
            time.sleep(3)  # Faster refresh
    except KeyboardInterrupt:
        print("\n\n👋 TPU Monitor stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()