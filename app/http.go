package main;

import "C"

import (
   "net/http"
)

func main() {}
//go build -o libhttp.so -buildmode=c-shared http.go
//go build -o libhttp.a -buildmode=c-archive http.go

//export Notify
func Notify(msg string){
    http.Get("http://47.115.166.195:9999/api/v1/ws/notify?msg="+msg)
}

