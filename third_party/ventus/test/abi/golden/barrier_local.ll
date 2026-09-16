; ModuleID = 'third_party/ventus/test/abi/barrier_local.cl'
source_filename = "third_party/ventus/test/abi/barrier_local.cl"
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128-A5-G1"
target triple = "riscv32"

@barrier_local.values = internal unnamed_addr addrspace(3) global [64 x float] undef, align 4

; Function Attrs: convergent norecurse nounwind vscale_range(1,2048)
define dso_local ventus_kernel void @barrier_local(ptr addrspace(1) nocapture noundef readonly align 4 %0, ptr addrspace(1) nocapture noundef writeonly align 4 %1) local_unnamed_addr #0 !kernel_arg_addr_space !6 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_base_type !8 !kernel_arg_type_qual !9 {
  %3 = call i32 @_Z12get_local_idj(i32 noundef 0) #3
  %4 = call i32 @_Z13get_global_idj(i32 noundef 0) #3
  %5 = getelementptr inbounds float, ptr addrspace(1) %0, i32 %4
  %6 = load float, ptr addrspace(1) %5, align 4, !tbaa !10
  %7 = getelementptr inbounds [64 x float], ptr addrspace(3) @barrier_local.values, i32 0, i32 %3
  store float %6, ptr addrspace(3) %7, align 4, !tbaa !10
  call void @llvm.riscv.ventus.barrier(i32 1)
  %8 = add nsw i32 %3, 1
  %9 = call i32 @_Z14get_local_sizej(i32 noundef 0) #3
  %10 = urem i32 %8, %9
  %11 = getelementptr inbounds [64 x float], ptr addrspace(3) @barrier_local.values, i32 0, i32 %10
  %12 = load float, ptr addrspace(3) %11, align 4, !tbaa !10
  %13 = getelementptr inbounds float, ptr addrspace(1) %1, i32 %4
  store float %12, ptr addrspace(1) %13, align 4, !tbaa !10
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare dso_local i32 @_Z12get_local_idj(i32 noundef) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare dso_local i32 @_Z13get_global_idj(i32 noundef) local_unnamed_addr #1

; Function Attrs: nounwind
declare void @llvm.riscv.ventus.barrier(i32 immarg) #2

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare dso_local i32 @_Z14get_local_sizej(i32 noundef) local_unnamed_addr #1

attributes #0 = { convergent norecurse nounwind vscale_range(1,2048) "disable-tail-calls"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="ventus-gpgpu" "target-features"="+32bit,+a,+m,+relax,+zdinx,+zfinx,+zhinx,+zve32f,+zve32x,+zvl32b,-64bit,-save-restore" "uniform-work-group-size"="false" }
attributes #1 = { convergent mustprogress nofree nounwind willreturn memory(none) "disable-tail-calls"="true" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="ventus-gpgpu" "target-features"="+32bit,+a,+m,+relax,+zdinx,+zfinx,+zhinx,+zve32f,+zve32x,+zvl32b,-64bit,-save-restore" }
attributes #2 = { nounwind }
attributes #3 = { convergent nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!opencl.ocl.version = !{!4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"ilp32"}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 1, !"SmallDataLimit", i32 8}
!4 = !{i32 2, i32 0}
!5 = !{!"clang version 16.0.0 (https://github.com/THU-DSP-LAB/llvm-project.git d4f2063fe81cbbefda34da60d4cd5c46bce3d231)"}
!6 = !{i32 1, i32 1}
!7 = !{!"none", !"none"}
!8 = !{!"float*", !"float*"}
!9 = !{!"const", !""}
!10 = !{!11, !11, i64 0}
!11 = !{!"float", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C/C++ TBAA"}
