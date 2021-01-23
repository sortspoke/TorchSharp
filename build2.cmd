@call build.cmd build --configuration Release
rd /S /Q C:\Users\Jasper\.nuget\packages\torchsharp
dotnet pack --configuration Release