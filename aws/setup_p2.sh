#!/bin/bash


export region=(aws configure get region)
export ami="ami-971be8ee"
export instanceType="p2.xlarge"

. $(dirname "$0")/setup_instance.sh
